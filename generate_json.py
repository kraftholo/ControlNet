import os
import json
import shutil
from tqdm import tqdm

def find_images_defects(directory_to_search, root_dir, target_dir_path, source_dir_path, new_dir_path):
    results = []
    counter = 0
    name_tracker = {}  # To track the occurrence of image names and handle duplicates

    if not os.path.exists(new_dir_path):
        os.makedirs(new_dir_path)

    for image_name in tqdm(os.listdir(directory_to_search)):
        if os.path.isfile(os.path.join(directory_to_search, image_name)):
            base_name, extension = os.path.splitext(image_name)
            for dirpath, dirnames, filenames in os.walk(root_dir):
                if image_name in filenames:
                    # Handle duplicate names
                    if image_name in name_tracker:
                        name_tracker[image_name] += 1
                        modified_name = f"{base_name}_{name_tracker[image_name]}{extension}"
                    else:
                        name_tracker[image_name] = 0
                        modified_name = image_name

                    parts = os.path.basename(dirpath).split('_')
                    defect_type = '_'.join(parts[1:-1])
                    defect_level = parts[-1]
                    
                    # Update paths with the new directory
                    source_path = os.path.join(source_dir_path, modified_name)
                    target_path = os.path.join(target_dir_path, modified_name)
                    
                    # Copy image to the new directory with potentially modified name
                    shutil.copy(os.path.join(dirpath, image_name), os.path.join(new_dir_path, modified_name))
                    
                    prompt = f"Chicken with {defect_type} {defect_level}"
                    
                    results.append({
                        "source": source_path.replace("\\", "/"),
                        "target": target_path.replace("\\", "/"),
                        "prompt": prompt
                    })
                    counter += 1

    print(f"Processed {counter} images. Duplicates included if any.")

    with open(r'C:\Users\Rasmu\ControlNet\dataset_stuff\image_defects_results_with_duplicates.json', 'w') as f:
        json.dump(results, f, indent=4)

    return "Completed. Results saved to image_defects_results_with_duplicates.json."

# Example usage
directory_to_search = r"C:\Users\Rasmu\ControlNet\dataset_stuff\GCI_Front_100"
root_directory = r"C:\Users\Rasmu\ControlNet\dataset_stuff\Front_Defects_Labeled"
target_dir_path = r"C:\Users\Rasmu\ControlNet\dataset_stuff\GCI_Front_With_Duplications_100"
source_dir_path = r"C:\Users\Rasmu\ControlNet\dataset_stuff\EDGE_GCI_Front_With_Duplications_100_low_50_high_200"
new_dir_path = r"C:\Users\Rasmu\ControlNet\dataset_stuff\GCI_Front_With_Duplications_100"

find_images_defects(directory_to_search, root_directory, target_dir_path, source_dir_path, new_dir_path)
