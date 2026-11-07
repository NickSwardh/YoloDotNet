# <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/994287a9-556c-495f-8acf-1acae8d64ac0" height=24> Pose Estimation Demo
**Estimate human poses in an image and draw the keypoints**

This demo loads a YOLO26 pose estimation model, finds the people in a crosswalk scene, prints what it found,
and saves the image with each person's keypoints, the connections between them, a bounding box, label and confidence score drawn on it.

## ▶️ Run the demo

Before you start, export the models; see the [Demo README](../README.md#-quick-start). This demo runs on the CPU.

From the repository root:

```bash
dotnet run --project Demo/PoseEstimationDemo
```

The demo prints each detected person with its confidence, saves the result to `YoloDotNet_Results/PoseEstimation.jpg` on your desktop and opens that folder.

## 🧩 How it works

[`Program.cs`](./Program.cs) runs four steps:

**1. Load the model.** The demo loads `yolo26s-pose.onnx` from the repository's `test/assets/Models` folder.
If the models are missing, run `python tools/download_models.py` from the repository root to download and export them into that folder.
The execution provider decides where inference runs; this demo uses the CPU.

```csharp
using var yolo = new Yolo(new YoloOptions
{
    ExecutionProvider = new CpuExecutionProvider(SharedConfig.GetTestModelV26(ModelType.PoseEstimation))
});
```

**2. Run inference.** `RunPoseEstimation` returns a list of `PoseEstimation` results, each with a label, a confidence,
a bounding box and the person's keypoints (`KeyPoints`).

```csharp
using var image = SKBitmap.Decode(SharedConfig.GetTestImage(ImageType.PoseEstimation));
var results = yolo.RunPoseEstimation(image, confidence: 0.25, iou: 0.7);
```

**3. Draw the results** on the image. `PoseKeyPoints` defines how the keypoints are colored and connected;
the demo loads them from `pose-keypoints.json`, which the project copies to the output folder.

```csharp
image.Draw(results, new PoseDrawingOptions
{
    PoseKeyPoints = PoseKeyPoints.Load(Path.Combine(AppContext.BaseDirectory, "pose-keypoints.json"))
});
```

**4. Save the image** as a JPEG to the output folder.

`SharedConfig` supplies the model and the test image (`test/assets/Media/crosswalk.jpg`), so the demo runs without any paths to set.
In your own code, pass the path to your model and image instead.

## ⚙️ Options

### Execution provider

```csharp
new CpuExecutionProvider(model)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | – | Path to the ONNX model, or the model as a `byte[]`. |

To run the demo on a GPU or another accelerator, see [Execution providers](../README.md#-execution-providers).

### Model settings (`YoloOptions`)

| Property | Default | Description |
|----------|---------|-------------|
| `ImageResize` | `Proportional` | `Proportional` keeps the aspect ratio and pads the image; `Stretched` resizes it to the model's input size. Use the mode your model was trained with. |
| `SamplingOptions` | Bilinear | How the image is resized. The default matches how YOLO models are trained. |

### Pose estimation (`RunPoseEstimation`)

| Parameter | Default | This demo | Description |
|-----------|---------|-----------|-------------|
| `confidence` | `0.2` | `0.25` | Minimum confidence for a person to be returned. |
| `iou` | `0.7` | `0.7` | Overlap threshold for removing duplicate boxes (non-maximum suppression). Lower values remove more overlapping boxes. |
| `roi` | `null` | – | Optional region of interest (`SKRectI`); only people inside it are detected. |

📖 For guidance on choosing these values, see the [Accuracy & Configuration Guide](../../AccuracyAndConfiguration.md).

### Drawing (`PoseDrawingOptions`)

The demo sets `PoseKeyPoints` and uses the defaults for everything else. For example, to hide the boxes and show only confident keypoints:

```csharp
image.Draw(results, new PoseDrawingOptions
{
    PoseKeyPoints = PoseKeyPoints.Load("pose-keypoints.json"),
    PoseConfidence = 0.8,
    DrawBoundingBoxes = false
});
```

| Property | Default | Description |
|----------|---------|-------------|
| `PoseKeyPoints` | `null` | How keypoints are colored and connected, loaded with `PoseKeyPoints.Load(path)`. Without it, keypoints are drawn as plain dots in `DefaultPoseColor`, and only when boxes, labels and tails are all turned off. |
| `PoseConfidence` | `0.65` | Minimum confidence for a keypoint to be drawn. |
| `DefaultPoseColor` | Yellow (`#FFF633`), semi-transparent | Color for keypoints and connections that `PoseKeyPoints` doesn't give a color. |
| `DrawBoundingBoxes` | `true` | Draw a box around each person. |
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

`KeyPointMarkers` is obsolete and has no effect; use `PoseKeyPoints` instead.

Tracked people can also be drawn with a tail that shows their path (`DrawTrackedTail`, `TailThickness`, `TailPaintColorStart`, `TailPaintColorEnd`).
Tails need tracking; see the [VideoStreamDemo](../VideoStreamDemo).

> [!TIP]
> `Draw` draws directly on an `SKBitmap`. If you use an `SKImage`, `Draw` returns a new `SKBitmap` with the results:
> `using var result = image.Draw(results, options);`

## 📚 See also

- [Demo README](../README.md): setup and the steps all demos share
- [Accuracy & Configuration Guide](../../AccuracyAndConfiguration.md): resize mode, sampling and thresholds
- [`pose-keypoints.json`](../../test/assets/PoseKeyPoints/pose-keypoints.json): the keypoint colors and connections the demo uses
