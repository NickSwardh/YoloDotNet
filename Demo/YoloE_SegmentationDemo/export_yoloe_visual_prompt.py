import numpy as np

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

# Load the text/visual YoloE model (e.g., medium segmentation variant)
# Available YoloE segmentation models:
# https://docs.ultralytics.com/models/yoloe/#available-models-supported-tasks-and-operating-modes
model = YOLOE("yoloe-11m-seg.pt")

# Define visual prompts
visual_prompts = dict(
    bboxes=np.array(
        [
            [352, 185, 375, 207] # Bounding box in [x, y, x + width, y + height] format
        ]),
    cls=np.array(
        [
            0 # Class id to be assigned the bounding box
        ])
)

# Run prediction on a different image, using reference image to guide what to look for
results = model.predict(
    "target_image.jpg",                 # Target image for detection
    refer_image="visual_prompt.jpg",    # Reference image used to get visual prompts
    visual_prompts=visual_prompts,
    predictor=YOLOEVPSegPredictor,
    conf=0.1                            # The prompts may require a lower confidence value
)

# Show inference results
results[0].show()

# Export model to ONNX format with opset 17
model.export(format="onnx", device=0, opset=17)