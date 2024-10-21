from ultralytics import YOLO
import cv2
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque
import numpy as np

data_deque = {}
deepsort = None

def init_tracker():
    global deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
    deepsort = DeepSort(
        cfg_deep.DEEPSORT.REID_CKPT,
        max_dist=cfg_deep.DEEPSORT.MAX_DIST,
        min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
        max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
        max_age=cfg_deep.DEEPSORT.MAX_AGE,
        n_init=cfg_deep.DEEPSORT.N_INIT,
        nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
        use_cuda=True,
    )

def xyxy_to_xywh(*xyxy):
    """"Calculates the relative bounding box from absolute pixel values."""
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def compute_color_for_labels(label):
    """Simple function that adds fixed color depending on the class."""
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    if label == 0: # Person
        color = (85, 45, 255)
    elif label == 2: # Car
        color = (222, 82, 175)
    elif label == 3:  # Motorbike
        color = (0, 204, 255)
    elif label == 5:  # Bus
        color = (0, 149, 255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def UI_box(x, img, color=None, label=None, line_thickness=None):
    """Draws a UI box around detected objects."""
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [np.random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def draw_boxes(img, bbox, names, object_id, identities=None, offset=(0, 0)):
    """Draws bounding boxes and object trails."""
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(object_id[i])
        obj_name = names[object_id[i]]
        label = f"{id}:{obj_name}"

        UI_box(box, img, label=label, color=color, line_thickness=2)

def predict():
    """Runs YOLOv8 with DeepSORT tracking on the video."""
    # Initialize the tracker
    init_tracker()

    # Load the YOLO model
    model = YOLO('yolov8n.pt')  # or your model file

    # Run inference on a video source
    results = model.predict(source="test3.mp4", show=True)

    # Get the results and draw bounding boxes
    for result in results:
        # Extract boxes and labels
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        names = model.names  # Object class names
        object_ids = result.boxes.cls.cpu().numpy()  # Object IDs
        
        # Process with DeepSORT and draw the boxes
        outputs = deepsort.update(boxes, result.boxes.conf.cpu().numpy(), object_ids, result.orig_img)
        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -2]
            object_id = outputs[:, -1]
            
            draw_boxes(result.orig_img, bbox_xyxy, names, object_id, identities)

if __name__ == "__main__":
    predict()
