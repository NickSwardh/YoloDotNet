# <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/994287a9-556c-495f-8acf-1acae8d64ac0" height=24> Batch Demo
**Process a folder of images in parallel**

This demo loads a YOLO26 object detection model and runs it on every image in a folder, several images at a time.
Each image is saved with bounding boxes, labels and confidence scores drawn on it.

## ▶️ Run the demo

Before you start, export the models and check the GPU requirements; see the [Demo README](../README.md#-quick-start).

1. In [`Program.cs`](./Program.cs), set `IMAGE_FOLDER` to a folder with your images. The folder should contain only images.
2. From the repository root, run:

   ```bash
   dotnet run --project Demo/BatchDemo
   ```

The demo saves each result as `ObjectDetection_<unique id>.jpg` in `YoloDotNet_Results/batch` on your desktop, prints how many images it processed and opens that folder.

## 🧩 How it works

[`Program.cs`](./Program.cs) runs three steps:

**1. Load the model.** The demo loads `yolo26s.onnx` from the repository's `test/assets/Models` folder.
If the models are missing, run `python tools/download_models.py` from the repository root to download and export them into that folder.
The execution provider decides where inference runs; this demo uses CUDA.

```csharp
using var yolo = new Yolo(new YoloOptions
{
    ExecutionProvider = new CudaExecutionProvider(SharedConfig.GetTestModelV26(ModelType.ObjectDetection))
});
```

**2. Find the images** in `IMAGE_FOLDER`.

**3. Detect, draw and save each image,** several at a time. `Parallel.ForEach` runs the loop on up to `MAX_PARALLEL_IMAGES` threads,
and all threads share the same `Yolo` instance:

```csharp
Parallel.ForEach(imagePaths, new ParallelOptions { MaxDegreeOfParallelism = MAX_PARALLEL_IMAGES }, imagePath =>
{
    using var image = SKBitmap.Decode(imagePath);
    var results = yolo.RunObjectDetection(image, confidence: 0.25);

    image.Draw(results);
    image.Save(Path.Combine(outputFolder, $"ObjectDetection_{Guid.NewGuid()}.jpg"), SKEncodedImageFormat.Jpeg, 80);
});
```

## ⚡ Parallel or sequential

**Parallel** (what this demo does) is the fastest way to process many images. While one image runs on the GPU,
other threads decode, draw and save images on the CPU, so the hardware stays busy.

- **Tune `MAX_PARALLEL_IMAGES`** for your machine. A handful of threads is usually enough; beyond that they mostly wait for the GPU.
  With 8 threads on an RTX 5070 Ti, the demo is about 4.5× faster than one image at a time
  (45 → 200 images per second, including decode, detection, drawing and saving).
- **Inference on one `Yolo` instance is thread-safe,** so all threads can share it. Anything else you use inside the loop must be thread-safe too.
- **Images finish in any order.** Shared state, such as a list of results, needs locking.

**Sequential** processing handles one image at a time. It's simpler and keeps the input order, which suits code that isn't thread-safe,
results that must stay in order, or machines with limited memory:

```csharp
foreach (var imagePath in imagePaths)
{
    using var image = SKBitmap.Decode(imagePath);
    var results = yolo.RunObjectDetection(image, confidence: 0.25);

    image.Draw(results);
    image.Save(Path.Combine(outputFolder, $"ObjectDetection_{Guid.NewGuid()}.jpg"), SKEncodedImageFormat.Jpeg, 80);
}
```

> [!IMPORTANT]
> Don't use object tracking (for example `SortTracker`) with parallel processing. A tracker follows objects from one frame to the next,
> so it needs the frames in order; on independent images processed in parallel its results are wrong.
> For tracking, see the [VideoStreamDemo](../VideoStreamDemo), which processes frames in order.

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

To run the demo on the CPU, or with another execution provider, see [Execution providers](../README.md#-execution-providers) in the Demo README.

### Model settings (`YoloOptions`)

| Property | Default | Description |
|----------|---------|-------------|
| `ImageResize` | `Proportional` | `Proportional` keeps the aspect ratio and pads the image; `Stretched` resizes it to the model's input size. Use the mode your model was trained with. |
| `SamplingOptions` | Bilinear | How the image is resized. The default matches how YOLO models are trained. |

### Detection (`RunObjectDetection`)

| Parameter | Default | This demo | Description |
|-----------|---------|-----------|-------------|
| `confidence` | `0.2` | `0.25` | Minimum confidence for an object to be returned. |
| `iou` | `0.7` | – | Overlap threshold for removing duplicate boxes (non-maximum suppression). Lower values remove more overlapping boxes. |
| `roi` | `null` | – | Optional region of interest (`SKRectI`); only objects inside it are detected. |

📖 For guidance on choosing these values, see the [Accuracy & Configuration Guide](../../AccuracyAndConfiguration.md).

### Batch settings (`Program.cs`)

| Constant | This demo | Description |
|----------|-----------|-------------|
| `IMAGE_FOLDER` | `path\to\image\folder` | The folder with the images to process. Set it before running the demo. |
| `MAX_PARALLEL_IMAGES` | `8` | How many images are processed at once (`ParallelOptions.MaxDegreeOfParallelism`). |

### Drawing (`DetectionDrawingOptions`)

`image.Draw(results)` uses the default drawing options. To change them, pass your own options:

```csharp
image.Draw(results, new DetectionDrawingOptions
{
    FontSize = 24,
    BoundingBoxHexColors = ["#00ff00", "#ff0000"],
    DrawConfidenceScore = false
});
```

Options can be shared by all threads, as long as you don't change them while the loop runs.
For every option and its default, see [Drawing](../ObjectDetectionDemo/README.md#drawing-detectiondrawingoptions) in the ObjectDetectionDemo README.

## 📚 See also

- [Demo README](../README.md): setup and the steps all demos share
- [ObjectDetectionDemo](../ObjectDetectionDemo/README.md): object detection on a single image, with all drawing options
- [VideoStreamDemo](../VideoStreamDemo): object detection on video, with tracking
- [Accuracy & Configuration Guide](../../AccuracyAndConfiguration.md): resize mode, sampling and thresholds
