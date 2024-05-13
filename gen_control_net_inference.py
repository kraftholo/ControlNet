from tqdm import tqdm
import json
import os
from configurationLoader import returnRepoConfig
repoConfig = returnRepoConfig("control_net_gen_cfg.yaml")
from share import *
import config
import os
import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


def read_json_file(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return data 


def infer_one_image(input_image, 
                    prompt, 
                    a_prompt, 
                    n_prompt, 
                    num_samples, 
                    image_resolution, 
                    ddim_steps, 
                    guess_mode, 
                    strength, 
                    scale, 
                    seed, 
                    eta,
                    model,
                    ddim_sampler
                ):
    
    with torch.no_grad():
        #print(f'input_image max: {input_image.max()}, input_image min: {input_image.min()}, input_image.shape: {input_image.shape}, input_image.dtype: {input_image.dtype}')
        #print()
        input_image = HWC3(input_image)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
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

# Model loading and preparation
model = create_model(repoConfig.inference.model_yaml_path).cpu()
model.load_state_dict(load_state_dict(repoConfig.inference.trained_controlnet_model, location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

control_net_captions = read_json_file(repoConfig.inference.control_net_captions_path)
print(f'Length of controlnetcaptions: {len(control_net_captions)}')

if(not os.path.exists(repoConfig.inference.output_path)):
    os.makedirs(repoConfig.inference.output_path)

# counter = 0
#Per image and it's caption info, create an inferred image
print("Starting inference on augmented images!")
for image_info in tqdm(control_net_captions):

    # if counter == 10:
    #     break
    # counter += 1

    segmented_img_path = image_info['source']
    segmented_image = cv2.imread(segmented_img_path)
    # segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
    
    inference_results = infer_one_image(
        input_image = segmented_image,
        prompt = image_info['prompt'],
         model = model,
        ddim_sampler=ddim_sampler,
        
        a_prompt = repoConfig.inference.controls.a_prompt,
        n_prompt = repoConfig.inference.controls.n_prompt,
        num_samples = repoConfig.inference.controls.num_samples,
        image_resolution = repoConfig.inference.controls.image_resolution,
        ddim_steps = repoConfig.inference.controls.ddim_steps,
        guess_mode = repoConfig.inference.controls.guess_mode,
        strength = repoConfig.inference.controls.strength,
        scale = repoConfig.inference.controls.scale,
        seed = repoConfig.inference.controls.seed,
        eta = repoConfig.inference.controls.eta
    )

    # Save the inferred images
    output_path = os.path.join(repoConfig.inference.output_path, os.path.basename(segmented_img_path))
    img_to_write = inference_results[0]
    img_to_write = cv2.cvtColor(img_to_write, cv2.COLOR_RGB2BGR)
    
    #Writing the first sample for now, it's modular for later
    cv2.imwrite(output_path,img_to_write)


