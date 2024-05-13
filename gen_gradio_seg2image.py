from share import *
import config
import os
import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.uniformer import UniformerDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from configurationLoader import returnRepoConfig
repoConfig = returnRepoConfig("control_net_gen_cfg.yaml")

from gen_gradio_seg2image_helper import instance_pool_pasting

apply_uniformer = UniformerDetector()

model = create_model(repoConfig.gradio.model_yaml_path).cpu()
model.load_state_dict(load_state_dict(repoConfig.gradio.trained_controlnet_model, location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    with torch.no_grad():
        #print(f'input_image max: {input_image.max()}, input_image min: {input_image.min()}, input_image.shape: {input_image.shape}, input_image.dtype: {input_image.dtype}')
        #print()
        input_image = HWC3(input_image)
        #detected_map = apply_uniformer(resize_image(input_image, detect_resolution))
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

# Call this function at the start to load your images
image_directory = repoConfig.gradio.default_img_dir
IMG_HEIGHT = 512

# print(f"Loading images from {image_directory}")
# image_dict = load_images_from_directory(image_directory)
# print(f"Loaded {len(image_dict)} images")

block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("# Control Stable Diffusion with Segmentation Maps")
    with gr.Row():
        with gr.Column():
            #input_image = gr.Image(source='upload', type="numpy")

            gr.Markdown("## Upload Image ")
            input_image = gr.Image(source='upload', type="numpy")
            image_name = gr.Textbox(label="Image Name", placeholder="ChosenImageNameXYZ.png (Enter Manually)")
            prompt = gr.Textbox(label="Prompt",placeholder= repoConfig.gradio.placeholder.prompt)
            image_directory = gr.Textbox(label="Image Directory", value=image_directory)
            paste_button = gr.Button(label="Paste Defects",icon='https://img.icons8.com/ios/452/paste.png',value="Paste Defects")
            run_button = gr.Button(label="Run",icon='https://img.icons8.com/ios/452/play.png',value="Run")
            
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                detect_resolution = gr.Slider(label="Segmentation Resolution", minimum=128, maximum=1024, value=512, step=1)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                eta = gr.Number(label="eta (DDIM)", value=0.0)
                a_prompt = gr.Textbox(label="Added Prompt", value= repoConfig.gradio.placeholder.a_prompt)
                n_prompt = gr.Textbox(label="Negative Prompt", value=repoConfig.gradio.placeholder.n_prompt)

        with gr.Column():
            gr.Markdown("## Pasted RGB Image")
            pasted_output = gr.Image(label='Pasted Output', show_label=False, height=IMG_HEIGHT)
            
            gr.Markdown("## Input Segmentation Mask")
            segmented_output = gr.Image(label='Segmented Output', show_label=False, height=IMG_HEIGHT)
            
            gr.Markdown("## Generated Image")
            result_gallery = gr.Gallery(label='Output', show_label=False, height=IMG_HEIGHT)
        
        # Setup the click events
            # Remove the incorrect assignment

        # len(output): 512, len(segmented_image): 512
        # pasted_output_image: (512, 512, 3) segmented_image: (512, 512, 3)
        # pasted_output_image: uint8 segmented_image: uint8

        paste_button.click(
            fn= instance_pool_pasting,
            inputs=[image_name, image_directory, prompt],
            outputs=[pasted_output, segmented_output]
        )

        
        ips = [segmented_output, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta]
        run_button.click(fn=process, inputs=ips, outputs=[result_gallery])


block.launch(server_name='0.0.0.0')
