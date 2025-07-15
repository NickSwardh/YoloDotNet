# YoloE Zero‑Shot Demo

This demo highlights **YoloE** — a powerful **zero-shot object detection and segmentation model** — now supported in **YoloDotNet v3.0**.  
YoloE lets you detect custom objects *by text prompt*, without needing custom training or datasets.

You can describe what you're looking for — and YoloE will find it.  
No retraining. No datasets. Just text prompts.

> **Note:** YoloDotNet currently only supports YoloE with **text prompts** (not visual prompts).

# Preparations
To use YoloE with text prompts in YoloDotNet, you’ll need to:

- Export a YoloE model to ONNX with your own custom prompts

This is required because YoloE relies on zero-shot text embeddings, which must be baked into the ONNX model before .NET inference.

# Export to ONNX

1. **Install Python & Ultralytics** - Follow the quickstart guide: [https://docs.ultralytics.com/quickstart](https://docs.ultralytics.com/quickstart/)

2. **Download a YoloE Text Prompt Model** - [Ultralytics YoloE Models](https://docs.ultralytics.com/models/yoloe/#available-models-supported-tasks-and-operating-modes)

3. **Update the Export Script**\
\
Open the included script `export_yoloe.py`, and customize your prompts:

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

4. **Run the Export**
    ```
    python export_yoloe.py
    ```
    The ONNX model will be saved in the same folder as the `.pt` model.


Boom, done!

# Using in YoloDotNet
Load the exported ONNX model just like any Segmentation model in YoloDotNet.

### Tip:
- You may need to lower the confidence threshold (e.g. 0.1–0.3) for reliable detections.
- Try different wording in prompts for better results.
- Try a larger YoloE model for better accuracy