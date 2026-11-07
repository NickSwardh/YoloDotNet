# <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/994287a9-556c-495f-8acf-1acae8d64ac0" height=24> Classification Demo
**Classify an image and draw the result**

This demo loads a YOLO26 classification model, classifies a photo of a hummingbird, prints the most likely class,
and saves the image with the class name and confidence drawn on it.

## ▶️ Run the demo

Before you start, export the models; see the [Demo README](../README.md#-quick-start). This demo runs on the CPU.

From the repository root:

```bash
dotnet run --project Demo/ClassificationDemo
```

The demo prints the class with its confidence, saves the result to `YoloDotNet_Results/Classification.jpg` on your desktop and opens that folder.

## 🧩 How it works

[`Program.cs`](./Program.cs) runs four steps:

**1. Load the model.** The demo loads `yolo26s-cls.onnx` from the repository's `test/assets/Models` folder.
If the models are missing, run `python tools/download_models.py` from the repository root to download and export them into that folder.
The execution provider decides where inference runs; this demo uses the CPU.

```csharp
using var yolo = new Yolo(new YoloOptions
{
    ExecutionProvider = new CpuExecutionProvider(SharedConfig.GetTestModelV26(ModelType.Classification))
});
```

**2. Run inference.** `RunClassification` returns a list of `Classification` results, each with a label and a confidence, most likely first.

```csharp
using var image = SKBitmap.Decode(SharedConfig.GetTestImage(ImageType.Classification));
var results = yolo.RunClassification(image, classes: 1);
```

**3. Draw the results** on the image, using the default drawing options.

```csharp
image.Draw(results);
```

**4. Save the image** as a JPEG to the output folder.

`SharedConfig` supplies the model and the test image (`test/assets/Media/hummingbird.jpg`), so the demo runs without any paths to set.
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

### Classification (`RunClassification`)

| Parameter | Default | This demo | Description |
|-----------|---------|-----------|-------------|
| `classes` | `1` | `1` | How many of the most likely classes to return. |

📖 For guidance on resize mode and sampling, see the [Accuracy & Configuration Guide](../../AccuracyAndConfiguration.md).

### Drawing (`ClassificationDrawingOptions`)

`image.Draw(results)` uses the defaults below. To change them, pass your own options:

```csharp
image.Draw(results, new ClassificationDrawingOptions
{
    FontSize = 24,
    FontColor = SKColors.Yellow,
    DrawLabelBackground = false
});
```

| Property | Default | Description |
|----------|---------|-------------|
| `Font` | System default | The label font. Use `SKTypeface.FromFamilyName("Arial")` for an installed font, or `SKTypeface.FromFile("path/to/font.ttf")`. |
| `FontSize` | `18` | Label font size. |
| `FontColor` | White | Label text color. |
| `EnableFontShadow` | `true` | Draw a shadow behind the label text. |
| `EnableDynamicScaling` | `true` | Scale the font size with the image resolution. |
| `DrawLabelBackground` | `true` | Draw a semi-transparent box behind the labels. |

The labels are drawn in the top-left corner of the image, one line per returned class.

> [!TIP]
> `Draw` draws directly on an `SKBitmap`. If you use an `SKImage`, `Draw` returns a new `SKBitmap` with the results:
> `using var result = image.Draw(results);`

## 📚 See also

- [Demo README](../README.md): setup and the steps all demos share
- [Accuracy & Configuration Guide](../../AccuracyAndConfiguration.md): resize mode and sampling
