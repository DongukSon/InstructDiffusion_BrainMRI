import os
import sys
import math
import random
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.multiprocessing as mp  # 멀티프로세싱 추가
from einops import rearrange
from omegaconf import OmegaConf
import scipy.io
import k_diffusion as K
from torch import autocast

sys.path.append("./stable_diffusion")
from stable_diffusion.ldm.util import instantiate_from_config

# =========================
input_dirs = [
    "/fast_storage/intern/data/instruction_tuning/Instruct_Pix2Pix/acceleration_mat/test/",
    # "/fast_storage/intern/data/instruction_tuning/Instruct_Pix2Pix/crossmodal_mat/test/",
    # "/fast_storage/intern/data/instruction_tuning/Instruct_Pix2Pix/denoising_mat/test/",
    # "/fast_storage/intern/data/instruction_tuning/Instruct_Pix2Pix/segmentation_mat/test/"
]
output_dir = "logs/v0126_9_acconly_outputs"
config_path = "configs/brain_mri_finetune.yaml"
ckpt_path = "checkpoints/v1-5-pruned-emaonly-brainmri_v0126_9_acconly.ckpt"
vae_ckpt = None
resolution = 512
steps = 50
cfg_text = 5.0
cfg_image = 1.25
seed = 42
# =========================

class CFGDenoiser(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = torch.cat([z] * 3)
        cfg_sigma = torch.cat([sigma] * 3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], cond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_txt_cond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return 0.5 * (out_img_cond + out_txt_cond) + \
            text_cfg_scale * (out_cond - out_img_cond) + \
            image_cfg_scale * (out_cond - out_txt_cond)

def load_model_from_config(config, ckpt, device):
    model = instantiate_from_config(config.model)
    pl_sd = torch.load(ckpt, map_location="cpu")
    if 'state_dict' in pl_sd:
        pl_sd = pl_sd['state_dict']
    model.load_state_dict(pl_sd, strict=False)
    model.to(device)
    model.eval()
    return model

def worker(rank, gpu_id, file_list):
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    
    config = OmegaConf.load(config_path)
    model = load_model_from_config(config, ckpt_path, device)
    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])

    print(f"[GPU {gpu_id}] Start processing {len(file_list)} files.")

    for mat_path in file_list:
        try:
            mat_data = scipy.io.loadmat(mat_path)
            input_image = Image.fromarray(mat_data['image']).convert("RGB")
            instr = mat_data['instruction']
            edit_instruction = str(instr[0]) if isinstance(instr, np.ndarray) and instr.shape == (1,) else str(instr)
            
            width, height = input_image.size
            factor = resolution / max(width, height)
            factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
            width_resize, height_resize = int((width * factor) // 64) * 64, int((height * factor) // 64) * 64
            input_image = ImageOps.fit(input_image, (width_resize, height_resize), method=Image.Resampling.LANCZOS)

            with torch.no_grad(), autocast("cuda"):
                cond = {"c_crossattn": [model.get_learned_conditioning([edit_instruction])]}
                img_tensor = (2 * torch.tensor(np.array(input_image)).float() / 255 - 1).to(device)
                img_tensor = rearrange(img_tensor, "h w c -> 1 c h w")
                cond["c_concat"] = [model.encode_first_stage(img_tensor).mode()]
                
                uncond = {"c_crossattn": [null_token], "c_concat": [torch.zeros_like(cond["c_concat"][0])]}
                sigmas = model_wrap.get_sigmas(steps)
                
                torch.manual_seed(seed if seed is not None else random.randint(0, 100000))
                z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
                z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, 
                                                      extra_args={"cond": cond, "uncond": uncond, 
                                                                 "text_cfg_scale": cfg_text, "image_cfg_scale": cfg_image})
                
                x = model.decode_first_stage(z)
                x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
                x = 255.0 * rearrange(x, "1 c h w -> h w c")
                
                edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
                # edited_image = ImageOps.fit(edited_image, (width, height), method=Image.Resampling.LANCZOS)
                
                # # 이진화: 그레이스케일 변환 후 검은색이 아닌 부분을 흰색으로 변환
                # edited_array = np.array(edited_image.convert('L'))  # 그레이스케일 변환
                # binary_array = (edited_array > 0).astype(np.uint8) * 255
                # edited_image = Image.fromarray(binary_array)
                
                base = os.path.splitext(os.path.basename(mat_path))[0]
                edited_image.save(os.path.join(output_dir, f"output_{base}.jpg"))
        except Exception as e:
            print(f"[ERROR] GPU {gpu_id} failed on {mat_path}: {e}")

def main():
    os.makedirs(output_dir, exist_ok=True)
    
    gpu_ids = list(range(torch.cuda.device_count()))
    num_gpus = len(gpu_ids)
    
    if num_gpus == 0:
        print("No GPU found.")
        return

    all_mat_files = []
    for input_dir in input_dirs:
        if os.path.isdir(input_dir):
            all_mat_files.extend([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.mat')])

    random.shuffle(all_mat_files)
    print(f"Total {len(all_mat_files)} files found. Distributing to {num_gpus} GPUs.")

    chunks = np.array_split(all_mat_files, num_gpus)

    processes = []
    mp.set_start_method('spawn', force=True)

    for i in range(num_gpus):
        p = mp.Process(target=worker, args=(i, gpu_ids[i], chunks[i].tolist()))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All inference tasks completed.")

if __name__ == "__main__":
    main()