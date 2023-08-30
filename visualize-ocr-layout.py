import json
import os
import cv2
from collections import defaultdict
from matplotlib import pyplot as plt

# Load the JSON file containing annotations
annotations_path = '/kaggle/input/11-generated-prediction-json-with-images/11-Generated-prediction-json-with-images/PipelineJSON'
annotations_files = [f for f in os.listdir(annotations_path) if f.endswith('.json')]

# Loop through the annotation files
for annotations_file in annotations_files:
    with open(os.path.join(annotations_path, annotations_file)) as f:
        annotations_data = json.load(f)
    
    image_filename = annotations_data['name']
    image_path = os.path.join('/kaggle/input/11-generated-prediction-json-with-images/11-Generated-prediction-json-with-images/PipelineImage', image_filename)
    
    image = cv2.imread(image_path)
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
    plt.imshow(image)
    plt.title(image_filename)
    plt.axis('off')
    plt.show()
