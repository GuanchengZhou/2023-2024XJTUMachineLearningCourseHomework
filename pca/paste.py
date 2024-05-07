import numpy as np
import cv2
import os
from PIL import Image
import glob

image_dir = "./"

images = []

image_files = glob.glob(os.path.join(image_dir, "result3", 'feature10', "*"))

image_size = 32
H, W = 5, 2

new_image = Image.new("RGB", (image_size*H, image_size*W), 'white')

for i, image_file in enumerate(image_files):
    print(i, image_file)
    image = Image.open(image_file).resize((image_size, image_size))
    images.append(image)
    y = (i)//H
    x = (i)%H

    new_image.paste(image, (x*image_size, y*image_size))

new_image.save(os.path.join(image_dir,'feature10.png'))