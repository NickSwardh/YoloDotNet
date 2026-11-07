# <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/994287a9-556c-495f-8acf-1acae8d64ac0" height=24> YOLO-E Segmentation Demo
**Find and segment objects you describe, with no training**

YOLO-E is a **zero-shot object detection and segmentation model**. Instead of training it on your own dataset,
you tell it what to find: describe the objects with **text prompts** ("red sports car"), or show it an example with a **visual prompt**
(an image and a bounding box). No retraining. No datasets. Just prompts that guide the model to the objects you're interested in.

This demo loads a YOLO-E model with your prompts built in, segments the matching objects in an image, prints what it found,
and saves the image with masks, bounding boxes, labels and confidence scores drawn on it.

## ▶️ Run the demo

This demo needs a YOLO-E model that you export yourself; `tools/download_models.py` doesn't include one.

1. Export a YOLO-E model with your prompts; see [Export a YOLO-E model](#-export-a-yolo-e-model) below.
2. In [`Program.cs`](./Program.cs), set `MODEL_PATH` to the exported `.onnx` file.
3. From the repository root, run:

   ```bash
   dotnet run --project Demo/YoloE_SegmentationDemo
   ```

The demo runs on the CPU. It prints each object it found with its confidence, saves the result to `YoloDotNet_Results/YoloE_Segmentation.jpg` on your desktop and opens that folder.

## 📦 Export a YOLO-E model

YOLO-E finds objects through text or visual embeddings, and YoloDotNet needs those embeddings **built into the ONNX model**.
So you choose your prompts first, then export the model with them.

### Install Ultralytics

1. **Install Python and Ultralytics:** follow the [Ultralytics quickstart guide](https://docs.ultralytics.com/quickstart/).
2. **Pick a YOLO-E model:** see the [available YOLO-E models](https://docs.ultralytics.com/models/yoloe/#available-models-supported-tasks-and-operating-modes). The scripts below use the medium segmentation model, `yoloe-11m-seg.pt`.

### Text prompts

With text prompts, you describe the objects to find in plain language: say "find red sports cars", and the model looks for them.

1. **Set your prompts** in the included script [`export_yoloe_text_prompt.py`](./export_yoloe_text_prompt.py):

   ```python
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
   ```

2. **Run the export:**

   ```bash
   python export_yoloe_text_prompt.py
   ```

   The ONNX model is saved next to the `.pt` model, ready to use in YoloDotNet.

### Visual prompts

With visual prompts, you show the model what to look for instead of naming it: "find more things that look like this".
You give it a reference image and a bounding box around an example of each object. Each bounding box is a separate class,
and you can give **one** example per class. The model then finds visually similar objects in other images.

**Bounding box format.** A visual prompt box is `[x1, y1, x2, y2]`: the top-left corner `(x1, y1)`, and the bottom-right corner `(x2, y2)`,
which is the top-left corner plus the box's width and height. For example, a box at x = 352, y = 185 with width 23 and height 22 is:

```
[352, 185, 352 + 23, 185 + 22] = [352, 185, 375, 207]
```

1. **Set your images and bounding boxes** in the included script [`export_yoloe_visual_prompt.py`](./export_yoloe_visual_prompt.py):

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
   ```

2. **Run the export:**

   ```bash
   python export_yoloe_visual_prompt.py
   ```

   The ONNX model is saved next to the `.pt` model, ready to use in YoloDotNet.

> [!WARNING]
> **Model License Notice:**
> YOLO-E models are **not part of YoloDotNet** or its MIT license. They are licensed by Ultralytics under **AGPL-3.0**.
> See [Ultralytics Licensing](https://www.ultralytics.com/license/).

## 🧩 How it works

In YoloDotNet, an exported YOLO-E model is a segmentation model like any other. [`Program.cs`](./Program.cs) runs four steps:

**1. Load the model** from `MODEL_PATH`. The execution provider decides where inference runs; this demo uses the CPU.

```csharp
using var yolo = new Yolo(new YoloOptions
{
    ExecutionProvider = new CpuExecutionProvider(MODEL_PATH),
    ImageResize = ImageResize.Stretched
});
```

**2. Run inference.** `RunSegmentation` returns a list of `Segmentation` results, each with a label, a confidence, a bounding box and a pixel mask.

```csharp
using var image = SKBitmap.Decode(SharedConfig.GetTestImage(ImageType.Segmentation));
var results = yolo.RunSegmentation(image, confidence: 0.20, pixelConfedence: 0.5);
```

**3. Draw the results** on the image, using the default drawing options.

```csharp
image.Draw(results);
```

**4. Save the image** as a JPEG to the output folder.

`SharedConfig` supplies the test image (`test/assets/Media/people.jpg`). In your own code, pass the path to your image instead.

## ⚙️ Options

### Execution provider

```csharp
new CpuExecutionProvider(model)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | – | Path to the ONNX model, or the model as a `byte[]`. |

To run the demo on a GPU, or with another execution provider, see [Execution providers](../README.md#-execution-providers) in the Demo README.

### Model settings (`YoloOptions`)

| Property | Default | This demo | Description |
|----------|---------|-----------|-------------|
| `ImageResize` | `Proportional` | `Stretched` | `Proportional` keeps the aspect ratio and pads the image; `Stretched` resizes it to the model's input size. Use the mode your model was trained with. |
| `SamplingOptions` | Bilinear | – | How the image is resized. The default matches how YOLO models are trained. |

### Segmentation (`RunSegmentation`)

| Parameter | Default | This demo | Description |
|-----------|---------|-----------|-------------|
| `confidence` | `0.2` | `0.20` | Minimum confidence for an object to be returned. |
| `pixelConfedence` | `0.65` | `0.5` | Minimum confidence for a pixel to belong to a mask. (The parameter is spelled this way in the API.) |
| `iou` | `0.7` | – | Overlap threshold for removing duplicate boxes (non-maximum suppression). Lower values remove more overlapping boxes. |
| `roi` | `null` | – | Optional region of interest (`SKRectI`); only objects inside it are detected. |

📖 For guidance on choosing these values, see the [Accuracy & Configuration Guide](../../AccuracyAndConfiguration.md).

### Drawing (`SegmentationDrawingOptions`)

`image.Draw(results)` uses the defaults below. To change them, pass your own options:

```csharp
image.Draw(results, new SegmentationDrawingOptions
{
    PixelMaskOpacity = 200,
    DrawContour = true,
    DrawBoundingBoxes = false
});
```

| Property | Default | Description |
|----------|---------|-------------|
| `DrawSegmentationPixelMask` | `true` | Fill each object's mask. |
| `PixelMaskOpacity` | `128` | Mask opacity, from 0 (transparent) to 255 (opaque). |
| `DrawContour` | `false` | Draw an outline around each mask. |
| `ContourThickness` | `2` | Outline thickness, in pixels. |
| `DrawBoundingBoxes` | `true` | Draw a box around each object. |
| `DrawLabels` | `true` | Draw the class name. |
| `DrawConfidenceScore` | `true` | Draw the confidence next to the label. |
| `DrawLabelBackground` | `true` | Draw a background behind the label. |
| `LabelPosition` | `Auto` | Where the label goes: `Auto` (picks a position that fits inside the image), `TopLeft`, `TopRight`, `BottomLeft` or `BottomRight`. |
| `BorderThickness` | `2` | Box border thickness, in pixels. |
| `BoundingBoxOpacity` | `128` | Box fill opacity, from 0 (transparent) to 255 (opaque). |
| `BoundingBoxHexColors` | Built-in palette | Box and mask colors as hex strings, one per class id. |
| `Font` | System default | The label font. Use `SKTypeface.FromFamilyName("Arial")` for an installed font, or `SKTypeface.FromFile("path/to/font.ttf")`. |
| `FontSize` | `18` | Label font size. |
| `FontColor` | White | Label text color. |
| `EnableFontShadow` | `true` | Draw a shadow behind the label text. |
| `EnableDynamicScaling` | `true` | Scale the font size and border thickness with the image resolution. |

Tracked objects can also be drawn with a tail that shows their path (`DrawTrackedTail`, `TailThickness`, `TailPaintColorStart`, `TailPaintColorEnd`).
Tails need tracking; see the [VideoStreamDemo](../VideoStreamDemo).

> [!TIP]
> `Draw` draws directly on an `SKBitmap`. If you use an `SKImage`, `Draw` returns a new `SKBitmap` with the results:
> `using var result = image.Draw(results);`

## 💡 Tips

- Prompt-based models often need a **lower confidence threshold**, for example 0.1–0.3.
- Try **different wording** in your text prompts for better results.
- Try a **larger YOLO-E model** for better accuracy.
- For troubleshooting and details on exporting to ONNX, see the [Ultralytics YOLO-E documentation](https://docs.ultralytics.com/models/yoloe/).

## 📚 See also

- [Demo README](../README.md): setup and the steps all demos share
- [Accuracy & Configuration Guide](../../AccuracyAndConfiguration.md): resize mode, sampling and thresholds
- [SegmentationDemo](../SegmentationDemo): segmentation with a standard YOLO model
