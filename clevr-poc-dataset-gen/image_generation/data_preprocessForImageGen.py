import os, sys, json
from collections import Counter
from pathlib import Path
file_path = Path(__file__).resolve()
path_current = file_path.parents[0]
path_root = file_path.parents[1]
sys.path.append(".")
sys.path.append(str(path_root))
sys.path.append(str(path_current))
import numpy as np
import random
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

#prepare data for image generative model pretraining
'''
[
   {
       "id": "000001",
        "image": "images/000001.png",
        "text": "A small red rubber sphere is left of a large blue metal cube."
   }
]
'''



def annotate_with_regions(src_images_train, destination_train_images):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_train_images):
        os.makedirs(destination_train_images)
    # Iterate over all images in the source directory
    for img_name in os.listdir(src_images_train):
        # Full path to the image
        img_path = os.path.join(src_images_train, img_name)

        # Check if it's an image file (you can extend this check if needed)
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Open the image
            image = Image.open(img_path)
            width, height = image.size
            # Draw grey lines dividing the image into four regions
            draw = ImageDraw.Draw(image)
            vertical_line_x = width // 2
            horizontal_line_y = height // 2

            # Grey color for the lines
            grey_color = (180, 180, 180)

            # Draw the lines
            draw.line([(vertical_line_x, 0), (vertical_line_x, height)], fill=grey_color, width=3)
            draw.line([(0, horizontal_line_y), (width, horizontal_line_y)], fill=grey_color, width=3)

            # Add text for region labels
            font = ImageFont.load_default()  # Using the default font

            # Coordinates for placing text at each corner (regions r0, r1, r2, r3)
            text_positions = {
                'r0': (5, 3),
                'r1': (width - 10, 3),
                'r2': (5, height - 10),
                'r3': (width-10, height-10)
            }

            # Add text to each region
            for region, position in text_positions.items():
                draw.text(position, region, fill=(255, 255, 255), font=font)

            # Save the annotated image
            annotated_image_path = os.path.join(destination_train_images, img_name)
            image.save(annotated_image_path)

            print(f"Annotated image saved to: {annotated_image_path}")
            



def find_region(x, y):
   
    image_width = 480
    image_height = 320
    
    REGIONS = {
    "r0": {"x": [0, 240], "y": [0, 160]}, 
    "r1": {"x": [240, 480], "y": [0, 160]},
    "r2": {"x": [0, 240], "y": [160, 320]},
    "r3": {"x": [240, 480], "y": [160, 320]}
    }
    
    for region_id, region in REGIONS.items():
        if region["x"][0] <= x < region["x"][1] and region["y"][0] <= y < region["y"][1]:
            return region_id
    
def getDesc(objects, relationships):
    text = ''
    num= len(objects)
    text = f'Generate a CLEVR style image with geometric shapes. The scene has 4 equally sized regions - r0 (top left), r1 (top right), r2 (bottom left) and r3 (bottom right). There are {num} objects in the scene.'
    id_obj = {}
    
    for idx, o in enumerate(objects):
        pos = o['pixel_coords']
        x=  pos[0]
        y = pos[1]
        region = find_region(x, y)
        text_obj = f'There is a {o["color"]} {o["size"]} {o["material"]} {o["shape"]} in region {region}.'
        id_obj[idx] = f'{o["color"]} {o["size"]} {o["material"]} {o["shape"]}'
        text = text + ' ' + text_obj
        
    for idx, rr in enumerate(relationships['right']): ## rr is the list of objects to the right of idx 
        if len(rr) == 0:
            continue
        text = text + ' ' + f'The objects to the right of {id_obj[idx]} are:'
        for i, o in enumerate(rr):
            if (i!=len(rr)-1):
                text = text + id_obj[o]+','
            else:
                text = text + id_obj[o]+'.'
        
    for idx, rr in enumerate(relationships['front']): ## rr is the list of objects to the right of idx 
        if len(rr) == 0:
            continue
        text = text + ' ' + f'The objects to the front of {id_obj[idx]} are:'
        for i, o in enumerate(rr):
            if (i!=len(rr)-1):
                text = text + ' '+ id_obj[o]+','
            else:
                text = text + ' '+ id_obj[o]+'.'
            
    return text    


def create_json_training(src_json, dest_json):
    dest_json_name = 'train.json'
    dest_json_path = os.path.join(dest_json, dest_json_name)
    # Load the CLEVR scenes JSON file
    with open(src_json, 'r') as file:
        data = json.load(file)
    scenes = data['scenes']
    image_data = []
    i=0
    for scene in scenes:
        scene_info ={}
        scene_info["id"] = scene['image_index']
        scene_info["image"] = scene['image_filename']
        scene_info["text"] = getDesc(scene['objects'], scene['relationships'])
        image_data.append(scene_info)
        print('Scene done:', i)
        i=i+1
        #break 
    
    # Write output JSON
    with open(dest_json_path, 'w') as outfile:
        json.dump(image_data, outfile, indent=4)

    print(f"JSON file created at: {dest_json_path}")

def main(args):
    destination_json = '/users/sbsh670/archive/clevr/CLEVR_v1.0/image_generative_model_training'
    src_json = '/users/sbsh670/archive/clevr/CLEVR_v1.0/scenes/CLEVR_train_scenes.json'
    destination_train_images = '/users/sbsh670/archive/clevr/CLEVR_v1.0/image_generative_model_training/images/train'
    src_images_train = '/users/sbsh670/archive/clevr/CLEVR_v1.0/images/train'
    src_images_val = '/users/sbsh670/archive/clevr/CLEVR_v1.0/images/val'
    src_images_test = '/users/sbsh670/archive/clevr/CLEVR_v1.0/images/test'
    #prepare training images
    #annotate_with_regions(src_images_train, destination_train_images)
    create_json_training(src_json, destination_json)
    #create json with image id, text description
    '''
    [
   {
       "id": "000001",
        "image": "images/000001.png",
        "text": "A small red rubber sphere is left of a large blue metal cube."
   }
   ]
    '''


if __name__ == "__main__":
    main(None)
