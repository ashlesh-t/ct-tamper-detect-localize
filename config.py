import os
import yaml
import numpy as np
import torch

def generate_random_mask(cube_shape, mask_size):
    z, y, x = cube_shape
    mask_z, mask_y, mask_x = mask_size
    z_start = np.random.randint(0, z - mask_z)
    y_start = np.random.randint(0, y - mask_y)
    x_start = np.random.randint(0, x - mask_x)
    return np.array([
        [z_start, z_start + mask_z],
        [y_start, y_start + mask_y],
        [x_start, x_start + mask_x]
    ])

# -----------------------------------------------
# Load parameters from YAML
# -----------------------------------------------

with open("parameters.yml", "r") as stream:
    params = yaml.safe_load(stream)

config = {}

# -----------------------------------------------
# DATA LOCATIONS
# -----------------------------------------------

config['healthy_scans_raw'] = params['healthy']['scans_raw']
config['healthy_coords'] = params['healthy']['coords_csv']
config['healthy_samples'] = params['healthy']['samples_file']
config['healthy_sample_chunks'] = params['healthy']['chunks']

config['unhealthy_scans_raw'] = params['unhealthy']['scans_raw']
config['unhealthy_coords'] = params['unhealthy']['coords_csv']
config['unhealthy_samples'] = params['unhealthy']['samples_file']
config['unhealthy_sample_chunks'] = params['unhealthy']['chunks']

# -----------------------------------------------
# OUTPUT / MODEL PATHS
# -----------------------------------------------

BASE_PATH = params['base_path']
config['modelpath_inject'] = params['modelpath_inject']
config['modelpath_remove'] = params['modelpath_remove']
config['progress'] = params['progress']

# -----------------------------------------------
# GAN / TRAINING SETTINGS
# -----------------------------------------------

for key in [
    'cube_shape', 'mask_xlims', 'mask_ylims', 'mask_zlims',
    'copynoise',
    'generator_filters', 'discriminator_filters',
    'use_residual_blocks', 'use_spectral_norm',
    'dropout', 'batch_norm_momentum', 'leaky_relu_slope',
    'conv_kernel_size', 'conv_stride', 'conv_padding',
    'final_kernel_size', 'final_stride',
    'lambda_pixel', 'lr_generator', 'lr_discriminator',
    'betas', 'loss_function', 'channels', 'num_classes',
    'isDataTrainChunks'
]:
    config[key] = params[key]

# Ensure numpy arrays where needed
config['cube_shape'] = np.array(config['cube_shape'])
config['mask_xlims'] = np.array(config['mask_xlims'])
config['mask_ylims'] = np.array(config['mask_ylims'])
config['mask_zlims'] = np.array(config['mask_zlims'])

# -----------------------------------------------
# DEVICE SETTINGS
# -----------------------------------------------

if torch.cuda.is_available():
    config['device'] = 'cuda'
    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    config['device'] = 'cpu'
    print("No GPU available, using CPU")
config['gpus'] = "0" if torch.cuda.is_available() else ""

# -----------------------------------------------
# Validation checks
# -----------------------------------------------

if config['mask_zlims'][1] > config['cube_shape'][0]:
    raise Exception('Out of bounds: cube mask is larger than cube on dimension z.')
if config['mask_ylims'][1] > config['cube_shape'][1]:
    raise Exception('Out of bounds: cube mask is larger than cube on dimension y.')
if config['mask_xlims'][1] > config['cube_shape'][2]:
    raise Exception('Out of bounds: cube mask is larger than cube on dimension x.')

# -----------------------------------------------
# Make Save Directories
# -----------------------------------------------

os.makedirs(config['modelpath_inject'], exist_ok=True)
os.makedirs(config['modelpath_remove'], exist_ok=True)
os.makedirs(config['progress'], exist_ok=True)
