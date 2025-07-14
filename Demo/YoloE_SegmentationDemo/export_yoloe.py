from ultralytics import YOLOE
import torch

# Optional: boost performance on GPUs
torch.backends.cuda.enable_flash_sdp(True)

# Load the text/visual YoloE model (e.g., medium segmentation variant)
# Available YoloE segmentation models:
# https://docs.ultralytics.com/models/yoloe/#available-models-supported-tasks-and-operating-modes
model = YOLOE("yoloe-11m-seg.pt")

# Define your target objects via text prompts
names = [
    "traffic light",
    "red sports car",
    "flying bird",
    "yellow construction helmet",
]

# Set classes and generate the associated text embeddings
model.set_classes(names, model.get_text_pe(names))

# Export model to ONNX format with opset 17
model.export(format="onnx", device=0, opset=17)