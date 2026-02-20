import torch

def load_sd(path):
    x = torch.load(path, map_location="cpu")
    return x["state_dict"] if "state_dict" in x else x

start = load_sd("./checkpoints/v1-5-pruned-emaonly-brainmri_v0126_5_cmonly.ckpt")
end   = load_sd("./checkpoints/v1-5-pruned-emaonly-brainmri_v0126_3_cmonly.ckpt")

keys = [
    "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight",
    "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight",
    "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight",
    "cond_stage_model.transformer.text_model.final_layer_norm.weight",
]

for k in keys:
    d = (end[k] - start[k]).abs()
    print(k)
    print("  max:", d.max().item(), "mean:", d.mean().item())


# import torch
# import os

# ckpt_path = "./checkpoints/v1-5-pruned-emaonly-brainmri_v0127_2_acconly_0.ckpt"

# if not os.path.isfile(ckpt_path):
#     raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}\n"
#                             "Please check the path and ensure the file exists.")
# print(f"Loading checkpoint from: {ckpt_path}")
# sd = torch.load(ckpt_path, map_location="cpu")

# if "state_dict" in sd:
#     sd = sd["state_dict"]


# target_key = "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"

# if target_key in sd:
#     param_tensor = sd[target_key]
#     print(f"\n[Target Key]: {target_key}")
#     print(f"[Tensor Shape]: {param_tensor.shape}")

#     first_values = param_tensor[0, :5] 
#     print(f"[First 5 values]:\n{first_values}")
# else:
#     print(f"\n[Error] Key '{target_key}' not found in state_dict.")
#     print("Available cond_stage_model keys (first 10):")
#     cond_keys = [k for k in sd.keys() if "cond_stage_model" in k]
#     for k in cond_keys[:10]:
#         print(f" - {k}")