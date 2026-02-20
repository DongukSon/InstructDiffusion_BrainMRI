import torch
import matplotlib.pyplot as plt
import yaml
from omegaconf import OmegaConf
import sys
import os

sys.path.insert(0, '/home/intern4/InstructDiffusion')
sys.path.insert(0, '/home/intern4/InstructDiffusion/stable_diffusion')
os.chdir('/home/intern4/InstructDiffusion')

from ldm.util import instantiate_from_config
from dataset.brain_mri.brain_mri import BrainMRIDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config_path = 'configs/brain_mri_finetune.yaml'
ckpt_path = 'checkpoints/v1-5-pruned-emaonly-brainmri_v0127_2_acconly_4.ckpt'

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
config = OmegaConf.create(config)

print("Loading model...")
model = instantiate_from_config(config.model)

sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
model.load_state_dict(sd, strict=False)
model.to(device)
model.eval()
print(f"Model loaded and weights restored from {ckpt_path}")


dataset = BrainMRIDataset(path='/fast_storage/intern/data/instruction_tuning/IP_Adapter/acceleration_mat/', split='val') 
one_batch = dataset[0]


batch = {
    "edited": one_batch["edited"].unsqueeze(0).to(device),
    "edit": {
        "c_concat": one_batch["edit"]["c_concat"].unsqueeze(0).to(device)
    }
}

@torch.no_grad()
def check_vae_bottleneck(model, batch):
    x_gt = batch["edited"]
    

    encoder_posterior = model.encode_first_stage(x_gt)
    z = model.get_first_stage_encoding(encoder_posterior)
    
    # VAE Decoding (Latent -> Pixel)
    x_rec = model.decode_first_stage(z)
    
    gt_img = (x_gt[0].cpu().permute(1, 2, 0) + 1) / 2
    rec_img = (x_rec[0].cpu().permute(1, 2, 0) + 1) / 2
    gt_img = torch.clamp(gt_img, 0, 1)
    rec_img = torch.clamp(rec_img, 0, 1)
    residual = torch.abs(gt_img - rec_img)
    
    return gt_img, rec_img, residual

gt, rec, res = check_vae_bottleneck(model, batch)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1); plt.imshow(gt); plt.title("Original Label")
plt.subplot(1, 3, 2); plt.imshow(rec); plt.title("VAE Reconstructed")
plt.subplot(1, 3, 3); plt.imshow(res.sum(-1), cmap='hot'); plt.title("Info Loss (Residual)")
import torch
import matplotlib.pyplot as plt
import yaml
from omegaconf import OmegaConf
import sys
import os

sys.path.insert(0, '/home/intern4/InstructDiffusion')
sys.path.insert(0, '/home/intern4/InstructDiffusion/stable_diffusion')
os.chdir('/home/intern4/InstructDiffusion')

from ldm.util import instantiate_from_config
from dataset.brain_mri.brain_mri import BrainMRIDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config_path = 'configs/brain_mri_finetune.yaml'
ckpt_path = 'checkpoints/v1-5-pruned-emaonly-brainmri_v0127_2_acconly_4.ckpt'

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
config = OmegaConf.create(config)

print("Loading model...")
model = instantiate_from_config(config.model)

sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
model.load_state_dict(sd, strict=False)
model.to(device)
model.eval()
print(f"Model loaded and weights restored from {ckpt_path}")


dataset = BrainMRIDataset(path='/fast_storage/intern/data/instruction_tuning/IP_Adapter/acceleration_mat/', split='val') 
one_batch = dataset[0]


batch = {
    "edited": one_batch["edited"].unsqueeze(0).to(device),
    "edit": {
        "c_concat": one_batch["edit"]["c_concat"].unsqueeze(0).to(device)
    }
}

@torch.no_grad()
def check_vae_bottleneck(model, batch):
    x_gt = batch["edit"]['c_concat']
    

    encoder_posterior = model.encode_first_stage(x_gt)
    z = model.get_first_stage_encoding(encoder_posterior)
    
    # VAE Decoding (Latent -> Pixel)
    x_rec = model.decode_first_stage(z)
    
    gt_img = (x_gt[0].cpu().permute(1, 2, 0) + 1) / 2
    rec_img = (x_rec[0].cpu().permute(1, 2, 0) + 1) / 2
    gt_img = torch.clamp(gt_img, 0, 1)
    rec_img = torch.clamp(rec_img, 0, 1)
    residual = torch.abs(gt_img - rec_img)
    
    return gt_img, rec_img, residual

gt, rec, res = check_vae_bottleneck(model, batch)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1); plt.imshow(gt); plt.title("Original Label")
plt.subplot(1, 3, 2); plt.imshow(rec); plt.title("VAE Reconstructed")
plt.subplot(1, 3, 3); plt.imshow(res.sum(-1), cmap='hot'); plt.title("Info Loss (Residual)")
plt.colorbar(fraction=0.046, pad=0.04)
plt.savefig('vae_result2.png') 
print("Result image saved as 'vae_result.png'")