# YoloE Zero‑Shot Demo

This demo highlights **YoloE** — a powerful **zero-shot object detection and segmentation model** — supported in YoloDotNet.

YoloE lets you detect custom objects either by **describing them with text prompts or by showing examples with visual prompts**, without needing any custom training or datasets.

You can simply tell the model what to find using natural language, or show it exactly what you’re looking for with an example image and bounding box.

No retraining. No datasets. Just prompts that guide the model to detect your objects of interest.

# Preparations
To use YoloE with text prompts in YoloDotNet, you’ll need to:

- Export a YoloE model to ONNX with your own custom prompts

This is required because YoloE relies on zero-shot text/visual embeddings, which must be baked into the ONNX model before .NET inference.

# Install Ultralytics
1. **Install Python & Ultralytics** - Follow the quickstart guide: [https://docs.ultralytics.com/quickstart](https://docs.ultralytics.com/quickstart/)

2. **Download a YoloE Text Prompt Model** - [Ultralytics YoloE Models](https://docs.ultralytics.com/models/yoloe/#available-models-supported-tasks-and-operating-modes)

# Text Prompts

YoloE supports zero-shot object detection using text prompts, allowing you to describe target objects using natural language — no retraining or datasets required.

It’s as simple as saying “find red sports cars” — and the model will detect them directly based on your input.

### Export Text Prompts to ONNX
You must embed your text prompts into the ONNX model before using it in YoloDotNet.

1. **Update the Export Script**\
Open the included script `export_yoloe_text_prompt.py`, and customize your prompts:

    ```python
    from ultralytics import YOLOE
    import torch

    # Optional: boost performance on some GPUs
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

    # Export model to ONNX format
    model.export(format="onnx", device=0, opset=17)
    ```

2. **Run the Export**
    ```
    python export_yoloe_text_prompt.py
    ```
    The ONNX model will be saved in the same folder as the `.pt` model, ready for use in YoloDotNet.

# Visual Prompts
In addition to using text labels to describe what you want the model to detect, YOLO‑E also supports visual prompts — a powerful feature that lets you show the model what to look for using an image instead of words.

Here’s how it works: you provide a reference image that contains an example of the object you’re interested in, along with a bounding box that highlights it. Each bounding box in the reference image corresponds to a different class. You can only provide one visual example per class. The model then uses these examples to search for visually similar objects in the input image.

It’s like saying, “find more things that look like this,” by pointing them out directly — no need to name them.

## Visual Prompt Bounding Box Format
Bounding box visual prompt format is `[x1, y1, x2, y2]`, which means:

- The box is specified by its top-left corner (x1, y1).
- The bottom-right corner (x2, y2) is calculated by adding the box’s width and height to the top-left corner coordinates.

For example, if you have a box with:

- x = 352
- y = 185
- Width = 23
- Height = 22

The correct bounding box for the visual prompt would be:
```
[352, 185, 352 + 23, 185 + 22] = [352, 185, 375, 207]
```

## Export visual prompts to ONNX

1. Open the included script `export_yoloe_visual_prompt.py`, and customize it with your images and bounding boxes:

    ```python
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
                0 # Class id to be assigned for the bounding box
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

    # Export the model for use in YoloDotNet
    model.export(format="onnx", device=0, opset=17)
    ```
2. **Run the Export**
    ```
    python export_yoloe_visual_prompt.py
    ```
    The ONNX model will be saved in the same folder as the `.pt` model, ready for use in YoloDotNet.

# Using in YoloDotNet
Load the exported ONNX model just like any other `Segmentation` model in YoloDotNet.

### Tip:
- You may need to lower the confidence threshold (e.g. 0.1–0.3) for reliable detections.
- Try different wording in text prompts for better results.
- Try a larger YoloE model for better accuracy
- Read the [Ultralytics YOLOE-documentation](https://docs.ultralytics.com/models/yoloe/)