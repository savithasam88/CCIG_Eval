import sys
from pathlib import Path
file_path = Path(__file__).resolve()
path_current = file_path.parents[0]
path_root = file_path.parents[1]
sys.path.append(".")
sys.path.append(str(path_root))
sys.path.append(str(path_current))

import random
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple

import cv2
import torch
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline

from skimage.draw import polygon as skpolygon
from scipy.spatial.distance import cdist

@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))




PROPERTIES = ['shape', 'color', 'material', 'size']
domain = {}
domain['color'] = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow'] 
domain['material'] = ['rubber', 'metal']
domain['shape'] = ['cube', 'cylinder', 'sphere', 'cone']
domain['size'] = ['large', 'small', 'medium']
region = [0,1,2,3]

CLEVR_COLORS_LIST = [
    'rgb: [87, 87, 87] .',      # gray
    'rgb: [173, 35, 35] .',     # red
    'rgb: [42, 75, 215] .',     # blue
    'rgb: [29, 105, 20] .',     # green
    'rgb: [129, 74, 25] .',     # brown
    'rgb: [129, 38, 192] .',    # purple
    'rgb: [41, 208, 208] .',    # cyan
    'rgb: [255, 238, 51] .'     # yellow
]

CLEVR_COLORS = {
    "gray": np.array([87,87,87]),
    "red": np.array([173,35,35]),
    "blue": np.array([42,75,215]),
    "green": np.array([29,105,20]),
    "brown": np.array([129,74,25]),
    "purple": np.array([129,38,192]),
    "cyan": np.array([41,208,208]),
    "yellow": np.array([255,238,51])
}

CLEVR_SIZES = {
    "large": 1,
    "medium": 0.58,
    "small": 0.39
}

REGIONS = {
    "0": {"x": [-240, 0], "y": [0, 160]}, #  {"x": [-5, 0], "y": [0, 5]},
    "1": {"x": [0.5, 240], "y": [0, 160]},
    "2": {"x": [-240, 0], "y": [-160, 0]},
    "3": {"x": [0.5, 240], "y": [-160, 0]}
}


def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon

def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert a polygon to a segmentation mask.

    Args:
    - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
    - image_shape (tuple): Shape of the image (height, width) for the mask.

    Returns:
    - np.ndarray: Segmentation mask with the polygon filled.
    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask

def load_image(image_str: str) -> Image.Image:
    if image_str.startswith("http"):
        image = Image.open(requests.get(image_str, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_str).convert("RGB")
    print('Image size:', image.size)
    size=(480, 320)
    # Resize the image to the desired size (320x240)
    image_resized = image.resize(size)

    return image

def get_boxes(results: DetectionResult) -> List[List[List[float]]]:
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)

    return [boxes]

def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks

def create_pipeline_and_processor(detector_id: Optional[str] = None, segmenter_id: Optional[str] = None):
    """
    Function to initialize the pipeline and processor once.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector_id = detector_id if detector_id is not None else "IDEA-Research/grounding-dino-tiny"
    segmenter_id = segmenter_id if segmenter_id is not None else "facebook/sam-vit-base"
    
    # Grounding DINO pipeline
    object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)

    # Segment Anything model and processor
    segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
    processor = AutoProcessor.from_pretrained(segmenter_id)

    return object_detector, segmentator, processor

def detect(
    image: Image.Image,
    labels: List[str],
    threshold: float = 0.7,
    object_detector: Optional = None #detector_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
    """
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    #detector_id = detector_id if detector_id is not None else "IDEA-Research/grounding-dino-tiny"
    #object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)

    labels = [label if label.endswith(".") else label+"." for label in labels]

    results = object_detector(image,  candidate_labels=labels, threshold=threshold)
    results = [DetectionResult.from_dict(result) for result in results]

    return results

def segment(
    image: Image.Image,
    detection_results: List[Dict[str, Any]],
    polygon_refinement: bool = False,
    segmentator: Optional = None,
    processor: Optional = None,
    
    #segmenter_id: Optional[str] = None
) -> List[DetectionResult]:
    """
    Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #segmenter_id = segmenter_id if segmenter_id is not None else "facebook/sam-vit-base"

    #segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
    #processor = AutoProcessor.from_pretrained(segmenter_id)

    boxes = get_boxes(detection_results)
    inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)

    outputs = segmentator(**inputs)
    masks = processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes
    )[0]

    masks = refine_masks(masks, polygon_refinement)

    for detection_result, mask in zip(detection_results, masks):
        detection_result.mask = mask

    return detection_results

def grounded_segmentation(
    image: Union[Image.Image, str],
    labels: List[str],
    threshold: float = 0.7,
    polygon_refinement: bool = False,
    detector: Optional = None,
    segmentator: Optional = None,
    processor: Optional = None
    #detector_id: Optional[str] = None,
    #segmenter_id: Optional[str] = None
) -> Tuple[np.ndarray, List[DetectionResult]]:
    if isinstance(image, str):
        image = load_image(image)

    detections = detect(image, labels, threshold, detector) #detector_id)
    detections = segment(image, detections, polygon_refinement, segmentator, processor)#segmenter_id)
    return np.array(image), detections




def annotate(image: Union[Image.Image, np.ndarray], detection_results: List[DetectionResult]) -> np.ndarray:
    # Convert PIL Image to OpenCV format
    image_cv2 = np.array(image) if isinstance(image, Image.Image) else image
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    # Iterate over detections and add bounding boxes and masks
    for detection in detection_results:
        label = detection.label
        score = detection.score
        box = detection.box
        mask = detection.mask

        # Sample a random color for each detection
        color = np.random.randint(0, 256, size=3)

        # Draw bounding box
        cv2.rectangle(image_cv2, (box.xmin, box.ymin), (box.xmax, box.ymax), color.tolist(), 2)
        cv2.putText(image_cv2, f'{label}: {score:.2f}', (box.xmin, box.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)

        # If mask is available, apply it
        if mask is not None:
            # Convert mask to uint8
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_cv2, contours, -1, color.tolist(), 2)

    return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

def plot_detections(
    image: Union[Image.Image, np.ndarray],
    detections: List[DetectionResult],
    save_name: Optional[str] = None
) -> None:
    annotated_image = annotate(image, detections)
    plt.imshow(annotated_image)
    plt.axis('off')
    if save_name:
        plt.savefig(save_name, bbox_inches='tight')
    plt.savefig('/users/sbsh670/archive/clevr/CLEVR_v1.0/images/val/0.png')
     

def random_named_css_colors(num_colors: int) -> List[str]:
    """
    Returns a list of randomly selected named CSS colors.

    Args:
    - num_colors (int): Number of random colors to generate.

    Returns:
    - list: List of randomly selected named CSS colors.
    """
    # List of named CSS colors
    named_css_colors = [
        'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond',
        'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
        'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey',
        'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
        'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue',
        'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite',
        'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory',
        'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow',
        'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray',
        'lightslategrey', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine',
        'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise',
        'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive',
        'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip',
        'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown',
        'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey',
        'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white',
        'whitesmoke', 'yellow', 'yellowgreen'
    ]

    # Sample random named CSS colors
    return random.sample(named_css_colors, min(num_colors, len(named_css_colors)))

def plot_detections_plotly(
    image: np.ndarray,
    detections: List[DetectionResult],
    class_colors: Optional[Dict[str, str]] = None
) -> None:
    # If class_colors is not provided, generate random colors for each class
    if class_colors is None:
        num_detections = len(detections)
        colors = random_named_css_colors(num_detections)
        class_colors = {}
        for i in range(num_detections):
            class_colors[i] = colors[i]


    fig = px.imshow(image)

    # Add bounding boxes
    shapes = []
    annotations = []
    for idx, detection in enumerate(detections):
        label = detection.label
        box = detection.box
        score = detection.score
        mask = detection.mask

        polygon = mask_to_polygon(mask)

        fig.add_trace(go.Scatter(
            x=[point[0] for point in polygon] + [polygon[0][0]],
            y=[point[1] for point in polygon] + [polygon[0][1]],
            mode='lines',
            line=dict(color=class_colors[idx], width=2),
            fill='toself',
            name=f"{label}: {score:.2f}"
        ))

        xmin, ymin, xmax, ymax = box.xyxy
        shape = [
            dict(
                type="rect",
                xref="x", yref="y",
                x0=xmin, y0=ymin,
                x1=xmax, y1=ymax,
                line=dict(color=class_colors[idx])
            )
        ]
        annotation = [
            dict(
                x=(xmin+xmax) // 2, y=(ymin+ymax) // 2,
                xref="x", yref="y",
                text=f"{label}: {score:.2f}",
            )
        ]

        shapes.append(shape)
        annotations.append(annotation)

    # Update layout
    button_shapes = [dict(label="None",method="relayout",args=["shapes", []])]
    button_shapes = button_shapes + [
        dict(label=f"Detection {idx+1}",method="relayout",args=["shapes", shape]) for idx, shape in enumerate(shapes)
    ]
    button_shapes = button_shapes + [dict(label="All", method="relayout", args=["shapes", sum(shapes, [])])]

    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        # margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        updatemenus=[
            dict(
                type="buttons",
                direction="up",
                buttons=button_shapes
            )
        ],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Show plot
    fig.show()

# Function to find closest color in CLEVR_COLORS
def closest_color(avg_color):
    avg_color = np.round(avg_color * 255).astype(np.uint8)
    # Calculate Euclidean distance between avg_color and all CLEVR colors
    color_distances = cdist([avg_color], list(CLEVR_COLORS.values()))
    closest_idx = np.argmin(color_distances)
    closest_color_name = list(CLEVR_COLORS.keys())[closest_idx]
    return closest_color_name

# Function to classify size based on the polygon area
def classify_size(area):
    if area < 500:
        return "small"
    elif area < 1500:
        return "medium"
    else:
        return "large"

# Function to classify the region based on bounding box center (optional)
def classify_region(x, y):
    image_width = 480
    image_height = 320
    
    x_mapped = (x/image_height)*10 - 5
    y_mapped = (y/image_width)*10 - 5
    for region_id, region in REGIONS.items():
        if region["x"][0] <= x_mapped <= region["x"][1] and region["y"][0] <= y_mapped <= region["y"][1]:
            return region_id
    return "Unknown"  # If it doesn't fit into any region

def geometry_shape(mask, area_bbox2):

    mask_uint8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return None
    circularity = 4 * np.pi * area / (perimeter**2 + 1e-6)

    # Sphere
    if circularity > 0.87:
        return "sphere"

    approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
    vertices = len(approx)

    # Cone (triangle)
    if vertices == 3:
        return "cone"

    elif vertices >= 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)

        if 0.85<aspect_ratio<1.15:
            return "cube"
        else:
            return "cylinder"

    return None

def estimate_size(bbox):
    x1, y1, x2, y2 = bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax
    area = (x2 - x1) * (y2 - y1)
    print('Area:', area, x2-x1, y2-y1)
    if (x2-x1)>60:
        return "large"
    return "small"

def bbox_iou(bbox1, bbox2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.
    
    Args:
    bbox1 (tuple): Bounding box 1 as (xmin, ymin, xmax, ymax)
    bbox2 (tuple): Bounding box 2 as (xmin, ymin, xmax, ymax)
    
    Returns:
    float: The IoU of the two bounding boxes
    """
    # Coordinates of the intersection area
    xmin_intersection = max(bbox1.xmin, bbox2.xmin)
    ymin_intersection = max(bbox1.ymin, bbox2.ymin)
    xmax_intersection = min(bbox1.xmax, bbox2.xmax)
    ymax_intersection = min(bbox1.ymax, bbox2.ymax)
    
    # If the boxes don't overlap, IoU is 0
    if xmin_intersection >= xmax_intersection or ymin_intersection >= ymax_intersection:
        return 0.0
    
    # Area of the intersection
    intersection_area = (xmax_intersection - xmin_intersection) * (ymax_intersection - ymin_intersection)
    
    # Area of both bounding boxes
    area_bbox1 = (bbox1.xmax - bbox1.xmin) * (bbox1.ymax - bbox1.ymin)
    area_bbox2 = (bbox2.xmax - bbox2.xmin) * (bbox2.ymax - bbox2.ymin)
    
    # Area of the union
    union_area = area_bbox1 + area_bbox2 - intersection_area
    
    # IoU: Intersection area / Union area
    iou = intersection_area / union_area
    
    return iou


def main(args):

    save = "/users/sbsh670/archive/clevr/CLEVR_v1.0/images/val/0.png"
    save_pol = "/users/sbsh670/archive/clevr/CLEVR_v1.0/images/val/0_pol"
    save_pol1 = "/users/sbsh670/archive/clevr/CLEVR_v1.0/images/val/0_pol1"
    image_url = "/users/sbsh670/archive/clevr/CLEVR_v1.0/images/val/CLEVR_val_000000.png"
    labels = ['sphere .', 'cube .', 'cylinder .']
    labels_size = ['size is relatively small .', 'size is large .']
    labels_shape = ['cube, all faces are squares, equal sides, sharp corners .', 'cylinder, round base, straight sides .', 'sphere, round in all directions, ball .'] 
    labels_color = ['color of object is gray .', 'color of object is red .', 'color of object is blue .', 'color of object is green .', 'color of object is brown .', 'color of object is purple .', 'color of object is cyan .', 'color of object is yellow .'] 
    labels_material = ['hard, reflective, cold to touch, rings when struck like metal .', 'flexible, soft, stretchy, bouncy, matte finish like rubber.']

    threshold = 0.3
    detector_id = "IDEA-Research/grounding-dino-tiny"
    segmenter_id = "facebook/sam-vit-base"
     
    detector, segmentator, processor = create_pipeline_and_processor(detector_id, segmenter_id)
    image_array, detections = grounded_segmentation(
    image=image_url,
    labels=labels,
    threshold=threshold,
    polygon_refinement=True,
    detector=detector,           
    segmentator=segmentator,     
    processor=processor)
    #plot_detections(image_array, detections, save)
    #find unique masks and corresponding bb
    bboxes = []
    
    print('Len of detection:', len(detections))
    for idx, detection in enumerate(detections):
        present = False
        p_s = 0
        p_b = None
        box = detection.box
        score = detection.score
        for bb_score in bboxes:
            bb = bb_score[0]
            if bbox_iou(bb, box)> 0.2:
                p_s = bb_score[1]
                p_b = bb
                present = True
                break
        
        if present:
            if p_s>=score:
                continue 
            else:
                bboxes.remove((p_b, p_s, idx))

        bboxes.append((box,score, idx))       

    idx_interested = []
    for bb in bboxes:
        idx_interested.append(bb[2])

    objects = []    
    for idx, detection in enumerate(detections):
        if idx not in idx_interested:
            continue
        
        label = detection.label
        box = detection.box
        score = detection.score
        mask = detection.mask
        '''
        present = False
        p_s = 0
        p_b = None
        for bb_score in bboxes:
            bb = bb_score[0]
            if bbox_iou(bb, box)> 0.2:
                p_s = bb_score[1]
                p_b = bb
                present = True
                break
        
        if present:
            if p_s>=score:
                continue 
            else:
                bboxes.remove((p_b, p_s))

        
        
        bboxes.append((box,score))       
        '''
        min_x = box.xmin
        min_y = box.ymin
        max_x = box.xmax
        max_y = box.ymax
        
        polygon = mask_to_polygon(mask)
        polygon = np.array(polygon)
        polygon_mask = np.zeros(image_array.shape[:2], dtype=np.uint8)
        rr, cc = skpolygon(polygon[:, 1], polygon[:, 0], polygon_mask.shape)
        polygon_mask[rr, cc] = 1
        masked_image = image_array * np.expand_dims(polygon_mask, axis=-1)
        masked_image_det = masked_image.astype(np.uint8)  # ensure correct dtype
        masked_image_det = Image.fromarray(masked_image_det)
        
        dino_shape_detections = detect(masked_image_det, labels_shape, threshold, detector)
        dino_color_detections = detect(masked_image_det, CLEVR_COLORS_LIST, threshold, detector)
        dino_material_detections = detect(masked_image_det, labels_material, threshold, detector)

        # Using max() to get the detection with the highest score
        det_shape = max(dino_shape_detections, key=lambda x: x.score, default=None)
        det_color = max(dino_color_detections, key=lambda x: x.score, default=None)
        det_mat = max(dino_material_detections, key=lambda x: x.score, default=None)
        #for idx, det in enumerate(dino_scm_detections):
        shape_label = det_shape.label
        shape_score = det_shape.score

        color_label = det_color.label
        color_score = det_color.score

        material_label = det_mat.label
        material_score = det_mat.score

        #color_label = scm_label.split(' ')[0]
        #material_label = scm_label.split(' ')[1]
        #shape_label = scm_label.split(' ')[2]

        
        masked_pixels = masked_image[polygon_mask == 1]
        masked_image_pil = np.clip(masked_image, 0, 255).astype(np.uint8)
        
        
        
        plt.imsave(save_pol1+'_'+str(idx)+'.png', masked_image_pil)
        plt.imsave(save_pol+'_'+str(idx)+'.png', polygon_mask, cmap = 'gray')
        
        # Shape: Calculate bounding box for aspect ratio calculation
        #shape_label = geometry_shape(polygon_mask, box)
        
        # Size: Calculate area from the polygon mask (using regionprops)
        size_label = estimate_size(box)


        

        #Calculate region classification based on bounding box center
        bbox_center_x = (min_x + max_x) / 2
        bbox_center_y = (min_y + max_y) / 2
        bbdet = []
        bbdet.append(bbox_center_x)
        bbdet.append(bbox_center_y)


        region_label = classify_region(bbox_center_x, bbox_center_y)

        objects.append({"pixel_coords": bbdet, "color": color_label, "material": material_label, "shape": shape_label, "size": size_label, "region": region_label})
        # Output the results
        #print(f"Detection {idx+1}:")
        #print(f"Label: {label}, Score: {score:.2f}")
        #print(f"Size: {label}, Shape: {shape_label}, Color: {color_label}, Material: {material_label}, Score: {shape_score} {color_score} {material_score} {score}")
        #print(f"Region: {region_label}")
        #print('BB centre:', bbox_center_x, bbox_center_y)
        
    print('Len of objects:', len(objects))
    scene_graph = {"image_id": image_url, "objects": objects}
    for o in objects:
        print(o)
        print('\n')



if __name__ == "__main__":
    main(None)



     
