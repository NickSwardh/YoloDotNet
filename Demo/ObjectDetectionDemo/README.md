# <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/994287a9-556c-495f-8acf-1acae8d64ac0" height=24> Object Detection Demo
**Detect objects in an image and draw the results**

This demo loads a YOLO26 object detection model, detects the objects in a street scene, prints what it found,
and saves the image with bounding boxes, labels and confidence scores drawn on it.

## ▶️ Run the demo

Before you start, export the models and check the GPU requirements; see the [Demo README](../README.md#-quick-start).

From the repository root:

```bash
dotnet run --project Demo/ObjectDetectionDemo
```

The demo prints each detected object with its confidence, saves the result to `YoloDotNet_Results/ObjectDetection.jpg` on your desktop and opens that folder.

## 🧩 How it works

[`Program.cs`](./Program.cs) runs four steps:

**1. Load the model.** The demo loads `yolo26s.onnx` from the repository's `test/assets/Models` folder.
If the models are missing, run `python tools/download_models.py` from the repository root to download and export them into that folder.
The execution provider decides where inference runs; this demo uses CUDA.

```csharp
using var yolo = new Yolo(new YoloOptions
{
    ExecutionProvider = new CudaExecutionProvider(SharedConfig.GetTestModelV26(ModelType.ObjectDetection))
});
```

**2. Run inference.** `RunObjectDetection` returns a list of `ObjectDetection` results, each with a label, a confidence and a bounding box.

```csharp
using var image = SKBitmap.Decode(SharedConfig.GetTestImage(ImageType.ObjectDetection));
var results = yolo.RunObjectDetection(image, confidence: 0.25, iou: 0.5);
```

**3. Draw the results** on the image, using the default drawing options.

```csharp
image.Draw(results);
```

**4. Save the image** as a JPEG to the output folder.

`SharedConfig` supplies the model and the test image (`test/assets/Media/street.jpg`), so the demo runs without any paths to set.
In your own code, pass the path to your model and image instead.

## ⚙️ Options

### Execution provider

```csharp
new CudaExecutionProvider(model, gpuId: 0, trtConfig: null, enableCudaGraph: false)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | – | Path to the ONNX model, or the model as a `byte[]`. |
| `gpuId` | `0` | The GPU to run on. |
| `trtConfig` | `null` | TensorRT settings; see the [TensorRTDemo](../TensorRTDemo/README.md). |
| `enableCudaGraph` | `false` | CUDA Graph replay for faster inference; see the [CUDA execution provider README](../../YoloDotNet.ExecutionProvider.Cuda/README.md). |

To run the demo on the CPU, or with another provider, see [Execution providers](../README.md#-execution-providers).

### Model settings (`YoloOptions`)

| Property | Default | Description |
|----------|---------|-------------|
| `ImageResize` | `Proportional` | `Proportional` keeps the aspect ratio and pads the image; `Stretched` resizes it to the model's input size. Use the mode your model was trained with. |
| `SamplingOptions` | Bilinear | How the image is resized. The default matches how YOLO models are trained. |

### Detection (`RunObjectDetection`)

| Parameter | Default | This demo | Description |
|-----------|---------|-----------|-------------|
| `confidence` | `0.2` | `0.25` | Minimum confidence for an object to be returned. |
| `iou` | `0.7` | `0.5` | Overlap threshold for removing duplicate boxes (non-maximum suppression). Lower values remove more overlapping boxes. |
| `roi` | `null` | – | Optional region of interest (`SKRectI`); only objects inside it are detected. |

📖 For guidance on choosing these values, see the [Accuracy & Configuration Guide](../../AccuracyAndConfiguration.md).

### Drawing (`DetectionDrawingOptions`)

`image.Draw(results)` uses the defaults below. To change them, pass your own options:

```csharp
image.Draw(results, new DetectionDrawingOptions
{
    FontSize = 24,
    BoundingBoxHexColors = ["#00ff00", "#ff0000"],
    DrawConfidenceScore = false
});
```

| Property | Default | Description |
|----------|---------|-------------|
| `DrawBoundingBoxes` | `true` | Draw a box around each object. |
| `DrawLabels` | `true` | Draw the class name. |
| `DrawConfidenceScore` | `true` | Draw the confidence next to the label. |
| `DrawLabelBackground` | `true` | Draw a background behind the label. |
| `LabelPosition` | `Auto` | Where the label goes: `Auto` (picks a position that fits inside the image), `TopLeft`, `TopRight`, `BottomLeft` or `BottomRight`. |
| `BorderThickness` | `2` | Box border thickness, in pixels. |
| `BoundingBoxOpacity` | `128` | Box fill opacity, from 0 (transparent) to 255 (opaque). |
| `BoundingBoxHexColors` | Built-in palette | Box colors as hex strings, one per class id. |
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

## 📚 See also

- [Demo README](../README.md): setup and the steps all demos share
- [Accuracy & Configuration Guide](../../AccuracyAndConfiguration.md): resize mode, sampling and thresholds
- [VideoStreamDemo](../VideoStreamDemo): object detection on video, with tracking
