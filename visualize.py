import json
import os
import cv2
from collections import defaultdict
from matplotlib import pyplot as plt
import numpy as np

# Load the JSON file containing annotations
annotations_path = 'PipelineOutput/'
annotations_files = [f for f in os.listdir(annotations_path) if f.endswith('.json')]

# write a function cut a cv2.imread image to tight boundary and return the cropped image as 1800*2700 pixels
def cut_image(image):
    # convert to grayscaleS
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # threshold to get just the signature
    retval, thresh_gray = cv2.threshold(gray, thresh=50, maxval=255, type=cv2.THRESH_BINARY)
    # find where the signature is and make a cropped region
    points = np.argwhere(thresh_gray==0) # find where the black pixels are
    points = np.fliplr(points) # store them in x,y coordinates instead of row,col indices
    x, y, w, h = cv2.boundingRect(points) # create a rectangle around those points
    x, y, w, h = x-10, y-10, w+20, h+20 # make the box a little bigger
    crop = image[y:y+h, x:x+w] # create a cropped region of the gray image
    return crop

# resize the image to 1800*2700 pixels
def resize_image(image):
    return cv2.resize(image, (1800, 2700))

# Loop through the annotation files
for annotations_file in annotations_files:
    with open(os.path.join(annotations_path, annotations_file)) as f:
        annotations_data = json.load(f)
    
    image_filename = annotations_data['name']
    image_path = os.path.join('Raw_Image_1/', image_filename)
    
    print(image_path)
    image = cv2.imread(image_path)
    image = cut_image(image)
    image = resize_image(image)

    if image is None:
        print(f"Failed to load image at {image_path}")
        continue

    for segment_id, segment_info in annotations_data['segments'].items():
        points = segment_info['points']
        
        # Extract the rectangle coordinates
        x_min = int(min(point['x'] for point in points))
        y_min = int(min(point['y'] for point in points))
        x_max = int(max(point['x'] for point in points))
        y_max = int(max(point['y'] for point in points))
        
        category_label = segment_info['type']  # Get the category label
        
        # Draw the rectangle
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Add the category label
        label_x = x_min  # x-coordinate of the bottom-left corner
        label_y = y_min - 10  # y-coordinate shifted upward for the label
        cv2.putText(image, category_label, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert the image from BGR to RGB for display with matplotlib
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Show the image with annotations
    plt.figure(figsize=(16,25))
    plt.imshow(image)
    plt.title(image_filename)
    plt.axis('off')
    plt.show()
