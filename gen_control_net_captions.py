from tqdm import tqdm
import json
import os
from configurationLoader import returnRepoConfig
repoConfig = returnRepoConfig("control_net_gen_cfg.yaml")

def read_json_file(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return data 


#if \controlNet_out doesnt exist, make it
if not os.path.exists(repoConfig.inferencePaths.control_net_inference_output):
    os.makedirs(repoConfig.inferencePaths.control_net_inference_output)

imgInputDirRoot = repoConfig.controlNetCaptions.gen_img_dir
inputImgDir = os.path.join(imgInputDirRoot, repoConfig.filePaths.img_sub_dir)
segmentedDir = os.path.join(imgInputDirRoot, repoConfig.filePaths.mask_sub_dir)

outputPath = repoConfig.controlNetCaptions.captions_output

controlNetInput = read_json_file(repoConfig.controlNetCaptions.instdiffinput_path)
print(f'Num of images in controlNetInput (instDiffInput): {len(controlNetInput)}')

# print(f'GT defect label : ')

controlNetCaptions = []

for img_name,data in tqdm(controlNetInput.items()):
    img_path = f"{inputImgDir}\\{img_name}"
    output_img_path = f"{segmentedDir}\\{img_name}"

    img_caption = "Chicken with"

    for caption in data['captions']:
        if caption.startswith(repoConfig.ihfoodSpecificInfo.dataset_defect_prefix):
            img_caption += f" {caption.split(' ')[0]}:{caption.split(' ')[1]}"
    
    # save the caption
    # source: segmented image, target: original image
    controlNetCaptions.append({
                        "source": output_img_path.replace("\\", "/"),
                        "target": img_path.replace("\\", "/"),
                        "prompt": img_caption
                    })
with open(outputPath, 'w') as f:
    json.dump(controlNetCaptions, f, indent=4)