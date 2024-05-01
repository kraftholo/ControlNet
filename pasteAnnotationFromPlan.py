# %% [markdown]
# #### Inference
# 
# 1. Inference on one image 
#     - We take some user input
#         - What defect(s) and level : Maybe in sstring format
#         - use script to output that
#     - We can modify the generated masks
#     - That can be used as input to models 
# 
# ### Data generation
# 0. WE ARE ONLY USING MASKS FROM REAL DATA
# 0. WE ARE ONLY PASTING ONTO REAL DATA
# 1. Take the distribution of defects and levels from instance pool
# 2. Rule 1
#     - 1 type of defect per limb
#     - Nothing on the broken limbs
# 3. Rule 2
#     - Case: GT_Defect:0 dominates and we choose a fixed number of samples per level 
#     - Forced method: choose number of samples such that GT_Defect:[0-5] are closest to equal sample size
#     - Limited to unique defects per limb. 
#         - Example: GT_Defect:= has 10000 samples, GT_Defect:[1-5] has 1000 each. The fixed number is 3000
#         - We start by sampling from the GT_Defect:0 samples until we have reached the fixed number (3000)
#         - that means that we have REPLACED 10000-3000 = 7000 samples from GT_Defect:0 across the GT_Defect:[1-5]
#         - 2000*5 = 10000 is the total number of missing samples, so the remaining (10000-7000 = 3000) is random sampled
#         - 
#     - Case: GT_Defect:[0-5] weighted probability based on amount of samples relative to largest sample size.
#         - Example: GT_Defect:0 has 100 and GT_Defect:5 has 5 make it 20 times less likely to choose GT_Defect:0
# 
#     
# 

# %%


# %%





# %%
# mask priority list
priorityList = [3,1,2,4,5,6,12,7,8,9,10,11]

# categories to color mapping
labelToRgbMapping = {
    0: (0, 0, 0),
    1:(50, 183, 250),
    2:(178, 80, 80),
    3:(255, 204, 51),
    4:(255, 96, 55),
    5:(129, 127, 129),
    6:(208, 1, 255),
    7:(204, 51, 102),
    8:(255, 53, 94),
    9:(52, 209, 183),
    10:(131, 224, 112),
    11:(170, 240, 209),
    12:(61, 61, 245),
}

labelToCategoryNameMapping = {
    1:"DrumLeft",
    2:"DrumRight",
    3:"Breast",
    4:"WingLeft",
    5:"WingRight",
    6:"BreastBruise",
    7:"DrumBruise",
    8:"WingBruise",
    9:"BreastSkinFlaw",
    10:"DrumSkinFlaw",
    11:"WingSkinFlaw",
    12:"BreastBurn",
}

# %%


    

# %%
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches
import json
from pycocotools.mask import decode
import pycocotools.mask as mask_util

from PIL import Image
import base64
from io import BytesIO
import numpy as np
import cv2
from tqdm import tqdm


def read_json_file(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return data 

def plot_image_with_bbox(image, bbox, title):
    fig, ax = plt.subplots(1)
    ax.imshow(image, cmap='gray')
    # Create a Rectangle patch
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)
    plt.title(title)
    plt.show()

def get_centroid(mask):
    moments = cv2.moments(mask)
    centroid_x = int(moments['m10'] / moments['m00'])
    centroid_y = int(moments['m01'] / moments['m00'])
    return (centroid_x, centroid_y)

# Calculate centroids relative to their bounding boxes
def get_relative_centroid(mask, bbox):
    centroid = get_centroid(mask)
    #print(f'preprocessedSrcMask centroid: {centroid}')
    relative_centroid_x = centroid[0] - bbox[0]
    relative_centroid_y = centroid[1] - bbox[1]
    return (relative_centroid_x, relative_centroid_y)

# Function to calculate overlap
def calculate_overlap(defect, limb):
    overlapPx = float(np.sum((defect & limb) > 0))
    if(np.sum(defect > 0) == 0):
        return 0
    percentageOverlap = overlapPx / float(np.sum(defect > 0))
    return percentageOverlap

def get_segmented_image(imgData, maskW = 512, maskH = 512):
    w,h = maskW,maskH
    segmentedImage = np.zeros((h, w, 3), dtype=np.uint8)
    categories = imgData["categories"]
    #displayCategories= [labelToCategoryNameMapping[cat] for cat in categories]
    #print(f'getSegmentedImage(): categories :{displayCategories}')
    iterationOrder = []
    for prio in priorityList:
        all_occur = [i for i, cat in enumerate(categories) if cat == prio]
        iterationOrder.extend(all_occur)
    for i in iterationOrder:
        split = imgData["captions"][i].split(' ')
        levelKey = None
        if(len(split) == 2):
            levelKey = split[1]
            if levelKey == '0':
                continue
        mask = decode(imgData["masks"][i])
        color = labelToRgbMapping[imgData["categories"][i]]
        segmentedImage[mask == 1] = color
    return segmentedImage

def calc_bbox_from_mask(mask):
    y_indices, x_indices = np.where(mask > 0)
    if y_indices.size == 0 or x_indices.size == 0:
        return (0, 0, 0, 0)  # No mask present
    else:
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        bbox = (x_min, y_min, (x_max - x_min), (y_max - y_min))
        return bbox

def decode_base64_to_cv2(base64_string):
    image_data = base64.b64decode(base64_string)
    image_buffer = BytesIO(image_data)
    pil_image = Image.open(image_buffer)
    rgb_image = np.array(pil_image)
    cv2_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    return cv2_image

def mask_2_rle(binary_mask):
    rle = mask_util.encode(np.array(binary_mask[...,None], order="F", dtype="uint8"))[0]
    rle['counts'] = rle['counts'].decode('ascii')
    return rle

def sample_instance(instPool,defect, key):
    instances = instPool[defect][key]
    return np.random.choice(instances)

def get_rgb_mask_to_paste(img_dir_root, angle, scale, best_mask, sample_instance):
    # Decode the 512x512x3 rgb mask
    source_image_name = sample_instance["image_name"]
    sourceDefectMask = decode(sample_instance["mask"])
    source_image_path = os.path.join(img_dir_root, source_image_name)
    srcRGBImg = cv2.imread(source_image_path)
    # Get scaled dimensions of both the source image and source image defect mask
    scaled_rgb_mask_w,scaled_rgb_mask_h = int(sourceDefectMask.shape[1]*scale),int(sourceDefectMask.shape[0]*scale)
    scaled_rgb_w, scaled_rgb_h = int(srcRGBImg.shape[1]*scale),int(srcRGBImg.shape[0]*scale)
    # Scale the source image and the source image defect mask
    scaledSrcImgDefectMask = cv2.resize(sourceDefectMask, (scaled_rgb_mask_w,scaled_rgb_mask_h), interpolation=cv2.INTER_LINEAR)
    scaledSrcImg = cv2.resize(srcRGBImg, (scaled_rgb_w,scaled_rgb_h), interpolation=cv2.INTER_LINEAR)
    defect_x, defect_y, _,_ = calc_bbox_from_mask(scaledSrcImgDefectMask)
    defect_centroid = get_centroid(scaledSrcImgDefectMask)
    best_mask_bbox = calc_bbox_from_mask(best_mask)
    dx= best_mask_bbox[0] - defect_x
    dy= best_mask_bbox[1] - defect_y
    M = cv2.getRotationMatrix2D((defect_centroid[0], defect_centroid[1]), angle, 1)  # Rotation around image center
    M[:, 2] += [dx, dy]  # Apply translation
    pasted_rgb_crop = cv2.warpAffine(scaledSrcImg, M, (best_mask.shape[0],best_mask.shape[1]))
    pasted_rgb_crop = cv2.cvtColor(pasted_rgb_crop, cv2.COLOR_BGR2RGB)    

    # fig, axs = plt.subplots(1, 3, figsize=(12, 6))  # 1 row, 2 columns

    # # Display the first image on the first axis
    # axs[0].imshow(srcRGBMask)
    # axs[0].set_title('Decoded RGB mask')
    # axs[0].axis('off')  # Turn off axis numbering

    # # Display the second image on the second axis
    # axs[1].imshow(placedSrcRGBMask)
    # axs[1].set_title('Scaled and placed RGB mask')
    # axs[1].axis('off')  # Turn off axis numbering

    # # Display the second image on the second axis
    # axs[2].imshow(pasted_rgb_crop)
    # axs[2].set_title('Rotated and translated RGB mask')
    # axs[2].axis('off')  # Turn off axis numbering

    # # Show the plot
    # plt.show()                          
    return pasted_rgb_crop

def get_scaled_and_translated_paste_sample(destLimbBbox, destLimbMask, preProcessedSrcMask, preProcessedSrcLimbBox,maskW,maskH, dest_image, img_dir_root, scale, sampleInstance):
    # Get the relative offset of the defect from the scaled (from before) mask and bbox
    relative_defect_centroid = get_relative_centroid(preProcessedSrcMask, preProcessedSrcLimbBox)
    #print(f'relative_defect_centroid: {relative_defect_centroid}')
    defect_centroid = get_centroid(preProcessedSrcMask)
    #print(f'defect_centroid: {defect_centroid}')

    #new_height, new_width = preProcessedSrcMask.shape[0], preProcessedSrcMask.shape[1]
    #print(f'new_height: {new_height}, new_width: {new_width}')

    # Calculate the absolute position on the destination limb
    defect_start_x = destLimbBbox[0] + relative_defect_centroid[0]
    defect_start_y = destLimbBbox[1] + relative_defect_centroid[1]
    #print(f'defect_start_x: {defect_start_x}, defect_start_y: {defect_start_y}')

    # Translate the defect mask to start at this new position
    translate_x = defect_start_x - defect_centroid[0]
    translate_y = defect_start_y - defect_centroid[1]
    #print(f'translate_x: {translate_x}, translate_y: {translate_y}')

    print(f'Relative defect centroid placement on destination limb: {defect_start_x, defect_start_y}, translated: {translate_x, translate_y} at scale: {round(scale,2)}')
    translation_matrix = np.float32([[1, 0, translate_x], [0, 1, translate_y]])
    translated_defect = cv2.warpAffine(preProcessedSrcMask, translation_matrix, (maskW, maskH))

    # Brute force search for best rotation and translation
    best_overlap = 0.0
    best_mask = translated_defect  # Set default to initial translated defect
    
    percentOfMovement = 2.5/100
    x_translation = int(percentOfMovement*maskW)
    y_translation = int(percentOfMovement*maskH)
    
    count = 0
    while(best_overlap < 1.0 and count < 50):
        angle = np.round(np.random.normal(0, 5)).astype(int)
        dx = np.round(np.random.normal(0, x_translation)).astype(int)
        dy = np.round(np.random.normal(0, y_translation)).astype(int)
        M = cv2.getRotationMatrix2D((defect_centroid[0], defect_centroid[1]), angle, 1)  # Rotation around image center
        M[:, 2] += [dx, dy]  # Apply translation
        rotated_translated_defect = cv2.warpAffine(translated_defect, M, (maskW, maskH))
        overlap = calculate_overlap(rotated_translated_defect, destLimbMask)
        if overlap > best_overlap:
            best_overlap = overlap
            best_mask = rotated_translated_defect
            if(overlap == 1.0):
                print(f'Complete overlap success! at count: {count}, angle: {angle}, dx: {dx}, dy: {dy}')
            else:
                print(f'Current best overlap : {overlap}')
        count += 1

    rgb_mask_to_paste = get_rgb_mask_to_paste(
        img_dir_root=img_dir_root,
        angle=angle,
        scale=scale,
        best_mask=best_mask,
        sample_instance=sampleInstance
    )
    final_mask = np.logical_and(best_mask, destLimbMask).astype(int)
    target_pasted_sample = dest_image#cv2.cvtColor(cv2.imread(dest_image_path), cv2.COLOR_BGR2RGB)
    target_pasted_sample[final_mask == 1] = rgb_mask_to_paste[final_mask == 1]
    # Calculate bounding box for the final mask
    final_mask_bbox = calc_bbox_from_mask(final_mask)


    # plot_image_with_bbox(final_mask, final_mask_bbox, "FinalMask")
    # # Create a figure and a set of subplots
    # fig, axs = plt.subplots(1, 3, figsize=(12, 6))  # 1 row, 2 columns
    # # Display the first image on the first axis
    # axs[0].imshow(rgb_mask_to_paste)
    # axs[0].set_title('Pasted RGB Mask')
    # axs[0].axis('off')  # Turn off axis numbering
    # # Display the second image on the second axis
    # axs[1].imshow(target_pasted_sample)
    # combined_mask_uint8 = final_mask.astype(np.uint8) * 255
    # combined_mask_rgb = np.stack((combined_mask_uint8,)*3, axis=-1)  # Stack it to create an RGB representation
    # axs[1].imshow(combined_mask_rgb, cmap='summer', alpha=0.2)
    # axs[1].set_title('Destination RGB Image')
    # axs[1].axis('off')  # Turn off axis numbering
    # # Display the second image on the second axis
    # axs[2].imshow(target_pasted_sample)
    # axs[2].set_title('Destination RGB Image')
    # axs[2].axis('off')  # Turn off axis numbering
    # # Show the plot
    # plt.show()

    return final_mask, final_mask_bbox, target_pasted_sample
    
def pre_process_inst_mask(img_data,sampledInstance):
    srcLimbBox = sampledInstance["limb_bbox"]
    limbName = sampledInstance["limb_name"]
    # We need to get the corresponding limb dimensions for the img_data
    idx = img_data["categoryNames"].index(limbName)
    destLimbBox = img_data["boxes"][idx]
    # Get the limb box dimensions for src and dest
    x_src_limb,y_src_limb,w_src_limb,h_src_limb = np.round(srcLimbBox).astype(int)
    _,_,w_dest_limb,h_dest_limb = np.round(destLimbBox).astype(int)
    scale = min(w_dest_limb/w_src_limb,h_dest_limb/h_src_limb)
    #print(f'scale : {scale}')

    # Get the defect bounding box
    defectBbox = sampledInstance["bbox"]
    x_inst,y_inst,w_inst,h_inst = np.round(defectBbox).astype(int)
    # Crop the defect mask
    binaryMask_inst = decode(sampledInstance["mask"]) # 512x512
    croppedMask_inst = binaryMask_inst[y_inst:y_inst+h_inst,x_inst:x_inst+w_inst]
    # Scale the defect mask
    scaled_mask_width, scaled_mask_height = int(w_inst * scale), int(h_inst * scale)
    scaled_mask = cv2.resize(croppedMask_inst, (scaled_mask_width, scaled_mask_height), interpolation=cv2.INTER_LINEAR)
    # Scale the origin point of the mask
    scaled_mask_x, scaled_mask_y = int(x_inst * scale), int(y_inst * scale)
    canvas = np.zeros_like(binaryMask_inst)
    canvas[scaled_mask_y:scaled_mask_y+scaled_mask_height, scaled_mask_x:scaled_mask_x+scaled_mask_width] = scaled_mask
    # Now we also scale the bounding box of the limb
    scaled_src_limbBox_x, scaled_src_limbBox_y = int(x_src_limb * scale), int(y_src_limb * scale)
    scaled_src_limbBox_width, scaled_src_limbBox_height = int(w_src_limb * scale), int(h_src_limb * scale)
    # Limb is also "moved"
    scaled_src_limb_box = [scaled_src_limbBox_x, scaled_src_limbBox_y, scaled_src_limbBox_width, scaled_src_limbBox_height]
    # Returning the scaled mask (cutout) and the scaled src limb box
    return canvas, scaled_src_limb_box, scale

def paste_sampled_instance(img_data, modified_img_data, sampledInstance, defectKey, levelKey, maskToPaste, bboxToPaste):
    # Ensure bboxToPaste is a list of native Python integers
    bbox_converted = [int(coord) for coord in bboxToPaste]

    # Define a common procedure to update data structures
    def update_data_structures(data_dict, mask, bbox, category, area, caption, category_name):
        data_dict["masks"].append(mask_2_rle(mask))
        data_dict["boxes"].append(bbox)
        data_dict["captions"].append(caption)
        data_dict["categories"].append(int(category))  # Explicit conversion to int
        data_dict["areas"].append(int(area))  # Explicit conversion to int
        data_dict["categoryNames"].append(category_name)

    captionFromSample = f'{defectKey} {levelKey}'
    area = int(np.sum(maskToPaste == 1))  # Sum and convert to int

    if len(modified_img_data) == 0:
        # Update img_data if modified_img_data is empty
        update_data_structures(img_data, maskToPaste, bbox_converted, sampledInstance["category"],
                               area, captionFromSample, sampledInstance["category_name"])
        
        # Assign updated img_data to modified_img_data
        modified_img_data.update({
            "masks": img_data["masks"],
            "boxes": img_data["boxes"],
            "captions": img_data["captions"],
            "categories": img_data["categories"],
            "areas": img_data["areas"],
            "categoryNames": img_data["categoryNames"]
        })
    else:
        # Update modified_img_data directly
        update_data_structures(modified_img_data, maskToPaste, bbox_converted, sampledInstance["category"],
                               area, captionFromSample, sampledInstance["category_name"])
        
    return modified_img_data

import re

def parse_defects(caption):
    # Extract the relevant part of the caption after 'with'
    defects_part = caption.split('with ')[1]

    # Use regular expression to match patterns like 'GT_DrumBruiseLeft:2'
    defects_list = re.findall(r'([A-Za-z_]+:\d+)', defects_part)

    # Initialize an empty dictionary to store the defects and their levels
    defects_dict = {}

    # Process each defect entry
    for defect in defects_list:
        # Split each entry by colon to separate the defect name and its level
        defect_name, level = defect.split(':')

        # Convert the level to an integer and add to the dictionary
        defects_dict[defect_name] = level

    return defects_dict

# %%
# import matplotlib.pyplot as plt

# import copy

# instDiffModified = {}
# img_dir_root = r'E:\thesis\datasets\images-512'

# count = 0 
# for image_name, data in instDiffInput.items():

#     if(count == 10):
#         break
#     count += 1
#     print(image_name)
#     # Image.fromarray(segmentedImage).save(output_img_path)
#     image_path = os.path.join(img_dir_root, image_name)
    
#     editedCaptions = [caption.split(' ')[0] for caption in data['captions']]
#     absentDefects = [defect for defect in defectKeys if defect not in editedCaptions]

#     categories = data["categories"]
#     displayCategories= [labelToCategoryNameMapping[cat] for cat in categories]
#     print(f'categories :{displayCategories}')
    
#     modified_image = {}
#     modifiedImageName = f'pasted_{image_name}'

#     for defect in absentDefects:
#         resize_flag = False
#         print(f'defect: {defect}')
#         keys = list(instPool[defect].keys())
#         # Filter out the key '0'
#         filtered_keys = [key for key in keys if key != '0']
#         levelKey = np.random.choice(filtered_keys)

#         sampledInstance = sampleInstance(instPool,defect,levelKey)
#         # print(f'sampleInstance: {sampledInstance["bbox"]}')
#         # print(f'sampleInstance: {sampledInstance["limb_bbox"]}')
#         limbName = sampledInstance["limb_name"]
#         idx = data["categoryNames"].index(limbName)
#         destLimbMask = decode(data["masks"][idx])    # 512x512
#         destLimbBbox = data["boxes"][idx]

#         maskW,maskH = data["masks"][0]["size"]

#         #plot_image_with_bbox(decode(sampledInstance["mask"]), sampledInstance["limb_bbox"], f'UnprocessedSrcMask w/ key: {levelKey}')
#         preProcessedSrcMask, preProcessedSrcLimbBox, scale = preProcessInstMask(data,sampledInstance)
#         finalMask, finalBbox, pastedRGBMask = returnBetterPaste(destLimbBbox,destLimbMask,preProcessedSrcMask,preProcessedSrcLimbBox,maskW,maskH,image_path,img_dir_root, scale,sampledInstance)
#         modifiedData = pasteSampledInstance2(data, modified_image,sampledInstance,defect,levelKey,finalMask,finalBbox)
    
#     instDiffModified[modifiedImageName] = modifiedData

#     #Save the modified image after pasting the defects
#     segmentedImage = getSegmentedImage(modifiedData,maskW,maskH)



# %% [markdown]
# ### Passing in the augmentation plan json

# %%

# Load all images from a directory into a dictionary
def load_images_from_directory(directory):
    image_dict = {}
    count = 0
    for filename in tqdm(os.listdir(directory)):
        # if(count == 10):
        #     break
        # count += 1
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Filter for image files
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path)
            if image is not None:
                image_dict[filename] = image
    return image_dict


def instance_pool_pasting(image_name, directory, prompt):
    imgDirRoot = directory
  
    image_path = os.path.join(imgDirRoot, image_name)
    dest_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    # print(f'image_dict_all: {image_dict_all.keys()}')

    instancePoolPath = r"E:\thesis\datasets\instancePoolAllTrainCombinedWithPredicted.json"
    instancePoolJson = read_json_file(instancePoolPath)

    instDiffJsonPath = r"E:\thesis\datasets\instDiffAllTrainCombinedWithPredicted.json"
    instDiffJson = read_json_file(instDiffJsonPath)

    instDiffAugmented = {}
    parsed_prompt = parse_defects(prompt)

    data = instDiffJson[image_name]

    print(f'Processing image: {image_name}')
    print(f'Processing data: {data}')
    
    augmented_image = {}
    #print(f'defect, level in augmentationPlanJson.items(): {augmentationPlanJson[0].items()}')
    for defect, level in parsed_prompt.items():
        print(f"Defect: {defect}, Level: {level}")
        print(f'\nPasting defect: {defect} at level: {level} on image: {image_name}')
        sampled_instance = sample_instance(instancePoolJson, defect,level)
        print(f'sampled_instance: {sampled_instance}')
        limbName = sampled_instance["limb_name"]
        idx = data["categoryNames"].index(limbName)
        destLimbMask = decode(data["masks"][idx]) 
        destLimbBbox = data["boxes"][idx]

        maskW,maskH = data["masks"][0]["size"]

        #plot_image_with_bbox(decode(sampledInstance["mask"]), sampledInstance["limb_bbox"], f'UnprocessedSrcMask w/ key: {levelKey}')
        preProcessedSrcMask, preProcessedSrcLimbBox, scale = pre_process_inst_mask(data,sampled_instance)
        binary_mask_to_paste, binary_mask_bbox, target_pasted_sample = get_scaled_and_translated_paste_sample(
                                                                                                           destLimbBbox=destLimbBbox,
                                                                                                           destLimbMask=destLimbMask,
                                                                                                           preProcessedSrcMask=preProcessedSrcMask,
                                                                                                           preProcessedSrcLimbBox=preProcessedSrcLimbBox,
                                                                                                           maskW=maskW,
                                                                                                           maskH=maskH,
                                                                                                           dest_image=dest_image,
                                                                                                           img_dir_root=imgDirRoot,
                                                                                                           scale=scale,
                                                                                                           sampleInstance=sampled_instance                                                                                                 
                                                                                                        )
        augmentedData = paste_sampled_instance(data, augmented_image, sampled_instance,defect,level,binary_mask_to_paste,binary_mask_bbox)

    instDiffAugmented[image_name] = augmentedData


    #Save the modified image after pasting the defects
    maskW,maskH = data["masks"][0]["size"]
    return target_pasted_sample, get_segmented_image(instDiffAugmented[image_name], maskW,maskH)

# Debug
# image_directory = "E:/thesis/datasets/all-train-images-512-front"
# input_image_demo = cv2.cvtColor(cv2.imread(r"E:\thesis\datasets\all-train-images-512-front\Cam=F-FN=20230502.115044.336732_White-VID=FullView-PID=-x=0000-y=0000-w=1088-h=1456.png"), cv2.COLOR_BGR2RGB)        
# pasted_output_image, segmented_image = instance_pool_pasting("Cam=F-FN=20230502.115044.336732_White-VID=FullView-PID=-x=0000-y=0000-w=1088-h=1456.png",image_directory , "Chicken with GT_DrumSkinFlawLeft:1, GT_WingSkinFlawLeft:1, GT_BreastBurn:1")
# print(f'len(output): {len(pasted_output_image)}, len(segmented_image): {len(segmented_image)}')
# print(f'pasted_output_image: {pasted_output_image.shape}', f'segmented_image: {segmented_image.shape}')
# print(f'pasted_output_image: {pasted_output_image.dtype}', f'segmented_image: {segmented_image.dtype}')

