
from tqdm import tqdm
import json
import os
from configurationLoader import returnRepoConfig
repoConfig = returnRepoConfig('control_net_cfg.yaml')

def read_json_file(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return data 
imgInputDirRoot = repoConfig.captions.image_dir
inputImgDir = os.path.join(imgInputDirRoot, repoConfig.filePaths.img_sub_dir)
segmentedDir = os.path.join(imgInputDirRoot, repoConfig.filePaths.mask_sub_dir)

outputPath = repoConfig.captions.output_dir

controlNetInput = read_json_file(repoConfig.captions.inst_diff_json)
print(f'Num of images in controlNetInput (instDiffInput): {len(controlNetInput)}')
controlNetCaptions = []

for img_name,data in tqdm(controlNetInput.items()):

    img_path = f"{inputImgDir}\\{img_name}"
    output_img_path = f"{segmentedDir}\\{img_name}"

    img_caption = "Chicken with"

    for caption in data['captions']:
        if caption.startswith('GT_'):
            img_caption += f" {caption.split(' ')[0]}:{caption.split(' ')[1]}"

    controlNetCaptions.append({
                        "source": output_img_path.replace("\\", "/"),
                        "target": img_path.replace("\\", "/"),
                        "prompt": img_caption
                    })

with open(outputPath, 'w') as f:
    json.dump(controlNetCaptions, f, indent=4)


