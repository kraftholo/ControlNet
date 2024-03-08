
import os
from tqdm import tqdm
import cv2 as cv

input_directory = r"C:\Users\Rasmu\ControlNet\dataset_stuff\GCI_Front_With_Duplications_100"
output_directory = r"C:\Users\Rasmu\ControlNet\dataset_stuff\EDGE_GCI_Front_With_Duplications_100_low_50_high_200"
lower_bound = 50
higher_bound = 200

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

for filename in tqdm(os.listdir(input_directory)):
    file_path = os.path.join(input_directory, filename)
    #print(f'File path = {file_path}')
    img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
    #print(f'Image shape = {img.shape}')
    edge_lord_image = cv.Canny(img, lower_bound, higher_bound)
    #print(f'Edge Lord Image shape = {edge_lord_image.shape}')
    #print(f'Output Image path = {os.path.join(output_directory, filename)}')
    cv.imwrite(os.path.join(output_directory, filename), edge_lord_image)
