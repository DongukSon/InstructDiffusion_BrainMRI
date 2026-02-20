#!/usr/bin/env python3
import torch
import yaml
from omegaconf import OmegaConf
import sys
import os
sys.path.insert(0, '/home/intern4/InstructDiffusion')
sys.path.insert(0, '/home/intern4/InstructDiffusion/stable_diffusion')
os.chdir('/home/intern4/InstructDiffusion')

from stable_diffusion.ldm.util import instantiate_from_config

# Load config
config_path = 'configs/brain_mri_finetune.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

config = OmegaConf.create(config)

# Load model
print("Loading model...")
model = instantiate_from_config(config.model)
print(f"Model loaded: {type(model).__name__}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
non_trainable_params = total_params - trainable_params

print("\n" + "="*70)
print("MODEL PARAMETERS")
print("="*70)
print(f"Total Parameters:         {total_params:>15,} ({total_params/1e6:.2f}M)")
print(f"Trainable Parameters:     {trainable_params:>15,} ({trainable_params/1e6:.2f}M)")
print(f"Non-trainable Parameters: {non_trainable_params:>15,} ({non_trainable_params/1e6:.2f}M)")
print(f"Trainable Ratio:          {trainable_params/total_params*100:>14.2f}%")
print("="*70)

# Breakdown by module
print("\nBREAKDOWN BY MODULE:")
print("-"*70)

module_params = {}
for name, param in model.named_parameters():
    module_name = name.split('.')[0]
    if module_name not in module_params:
        module_params[module_name] = {'total': 0, 'trainable': 0}
    
    param_count = param.numel()
    module_params[module_name]['total'] += param_count
    if param.requires_grad:
        module_params[module_name]['trainable'] += param_count

for module_name in sorted(module_params.keys()):
    total = module_params[module_name]['total']
    trainable = module_params[module_name]['trainable']
    ratio = trainable / total * 100 if total > 0 else 0
    print(f"  {module_name:<30} {total:>12,} ({trainable:>12,} trainable, {ratio:>5.1f}%)")

print("-"*70)
