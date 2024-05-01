from share import *
import config

import cv2
import einops
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import os
from tqdm import tqdm
import json

model = create_model('E:/thesis/repos/ControlNet/models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('E:/thesis/repos/ControlNet/model_checkpoints/run_0003 (less frequent logging)/epoch=4-step=7799.ckpt', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)


def process(input_image, prompt, a_prompt, n_prompt, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, num_samples = 1):
    with torch.no_grad():
        img = HWC3(input_image)
        
        H, W, C = img.shape

        detected_map = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        #if seed == -1:
        #    seed = random.randint(0, 65535)
        #seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return results

def read_json_file(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return data 

num_samples = 1
image_resolution = 512
detect_resolution = 512
ddim_steps = 20
guess_mode = False
strength = 1.0
scale = 9.0
seed = 1495398671
eta = 0.0
a_prompt = 'best quality, extremely detailed'
n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
output_generated_image_dir = r"E:\thesis\datasets\control_net_inference_output\gen_images"
output_visualizations_image_dir = r"E:\thesis\datasets\control_net_inference_output\gen_images_triple_stacked_visualizations"
control_net_json = read_json_file(r"E:\thesis\datasets\images-512-segmented\controlNetJson - Copy.json")
breaker = 0
seed_everything(seed)
for json_element in tqdm(control_net_json):

    input_image = cv2.imread(json_element['source'])
    rgb_input_image = HWC3(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    prompt = json_element['prompt']

    gen_image = process(rgb_input_image, prompt, a_prompt, n_prompt, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta)[0]

    gen_image = cv2.cvtColor(gen_image, cv2.COLOR_BGR2RGB)

    image_name = str(f'CN_gen_{os.path.basename(json_element["source"])}')
    output_string = f"{os.path.join(output_generated_image_dir, image_name)}".replace("\\", "/")
    cv2.imwrite(output_string, gen_image)

    original_image = cv2.imread(json_element['target'])
    #rgb_original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    visualization_image_name = str(f'CN_gen_viz_{os.path.basename(json_element["target"])}')
    combined_image = np.concatenate((original_image, input_image, gen_image), axis=1)
    viz_output_string = f"{os.path.join(output_visualizations_image_dir, visualization_image_name)}".replace("\\", "/")
    cv2.imwrite(viz_output_string, combined_image)