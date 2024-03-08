
import os
from tqdm import tqdm
import cv2 as cv

from annotator.uniformer import UniformerDetector
model_uniformer = UniformerDetector()

input_directory = r"C:\Users\Rasmu\ControlNet\dataset_stuff\GCI_Front_With_Duplications_100"
output_directory = r"C:\Users\Rasmu\ControlNet\dataset_stuff\SEGMENT_GCI_Front_With_Duplications_100"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

for filename in tqdm(os.listdir(input_directory)):
    file_path = os.path.join(input_directory, filename)
    img = cv.imread(file_path)
    result = model_uniformer(img)
    cv.imwrite(os.path.join(output_directory, filename), result)