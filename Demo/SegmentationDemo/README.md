# <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/994287a9-556c-495f-8acf-1acae8d64ac0" height=24> Segmentation Demo
**Segment the objects in an image and draw the masks**

This demo loads a YOLO26 segmentation model, finds the objects in a street scene, prints what it found,
and saves the image with a pixel mask, bounding box, label and confidence score drawn for each object.

## ▶️ Run the demo

Before you start, export the models and check the GPU requirements; see the [Demo README](../README.md#-quick-start).

From the repository root:

```bash
dotnet run --project Demo/SegmentationDemo
```

The demo prints each segmented object with its confidence, saves the result to `YoloDotNet_Results/Segmentation.jpg` on your desktop and opens that folder.

## 🧩 How it works

[`Program.cs`](./Program.cs) runs four steps:

**1. Load the model.** The demo loads `yolo26s-seg.onnx` from the repository's `test/assets/Models` folder.
If the models are missing, run `python tools/download_models.py` from the repository root to download and export them into that folder.
The execution provider decides where inference runs; this demo uses CUDA.
The image is resized with `ImageResize.Stretched`, which resizes it to the model's input size without keeping the aspect ratio.

```csharp
using var yolo = new Yolo(new YoloOptions
{
    ExecutionProvider = new CudaExecutionProvider(SharedConfig.GetTestModelV26(ModelType.Segmentation)),
    ImageResize = ImageResize.Stretched
});
```

**2. Run inference.** `RunSegmentation` returns a list of `Segmentation` results, each with a label, a confidence,
a bounding box and a pixel mask for the object.

```csharp
using var image = SKBitmap.Decode(SharedConfig.GetTestImage(ImageType.Segmentation));
var results = yolo.RunSegmentation(image, confidence: 0.25, pixelConfedence: 0.5, iou: 0.5);
```

**3. Draw the results** on the image, using the default drawing options.

```csharp
image.Draw(results);
```

**4. Save the image** as a JPEG to the output folder.

`SharedConfig` supplies the model and the test image (`test/assets/Media/people.jpg`), so the demo runs without any paths to set.
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

| Property | Default | This demo | Description |
|----------|---------|-----------|-------------|
| `ImageResize` | `Proportional` | `Stretched` | `Proportional` keeps the aspect ratio and pads the image; `Stretched` resizes it to the model's input size. Use the mode your model was trained with. |
| `SamplingOptions` | Bilinear | Bilinear | How the image is resized. The default matches how YOLO models are trained. |

### Segmentation (`RunSegmentation`)

| Parameter | Default | This demo | Description |
|-----------|---------|-----------|-------------|
| `confidence` | `0.2` | `0.25` | Minimum confidence for an object to be returned. |
| `pixelConfedence` | `0.65` | `0.5` | Minimum confidence for a pixel to belong to the object's mask. Lower values give larger masks. (The parameter is spelled this way in the API.) |
| `iou` | `0.7` | `0.5` | Overlap threshold for removing duplicate boxes (non-maximum suppression). Lower values remove more overlapping boxes. |
| `roi` | `null` | – | Optional region of interest (`SKRectI`); only objects inside it are segmented. |

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
| `DrawSegmentationPixelMask` | `true` | Fill each object's pixel mask. |
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

## 📚 See also

- [Demo README](../README.md): setup and the steps all demos share
- [Accuracy & Configuration Guide](../../AccuracyAndConfiguration.md): resize mode, sampling and thresholds
- [YoloE_SegmentationDemo](../YoloE_SegmentationDemo): zero-shot segmentation with text or visual prompts
