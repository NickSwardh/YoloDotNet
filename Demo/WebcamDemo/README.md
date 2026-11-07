# <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/994287a9-556c-495f-8acf-1acae8d64ac0" height=24> Webcam Demo
**Real-time object detection on a webcam feed**

This WPF app shows your webcam feed and, at the press of a button, runs a YOLO26 object detection model on every frame.
You can adjust the confidence threshold, track objects across frames and filter the results by label while it runs.

## ▶️ Run the demo

**Requirements:** Windows, a webcam, and an NVIDIA GPU with CUDA.
Before you start, export the models and check the GPU requirements; see the [Demo README](../README.md#-quick-start).

From the repository root:

```bash
dotnet run --project Demo/WebcamDemo
```

The window shows the live feed. Use the controls below it:

| Control | What it does |
|---------|--------------|
| Slider | Confidence threshold, from 0.20 to 0.99 (starts at 0.80). |
| **Track Objects** | Tracks objects across frames, giving each a stable id and a tail that shows its path. |
| **Enable Filtering** | Shows only people, cats and dogs. |
| **Start Detection** / **Stop Detection** | Turns object detection on and off. |

While detection runs, the top-right corner shows how long inference took for the last frame, and the frame rate that corresponds to.

## 🧩 How it works

[`MainWindow.xaml.cs`](./MainWindow.xaml.cs) does the work; [`MainWindow.xaml`](./MainWindow.xaml) defines the window and its controls.

**1. Load the model.** The demo loads `yolo26s.onnx` from the repository's `test/assets/Models` folder.
If the models are missing, run `python tools/download_models.py` from the repository root to download and export them into that folder.
The execution provider decides where inference runs; this demo uses CUDA.

```csharp
_yolo = new Yolo(new YoloOptions
{
    ExecutionProvider = new CudaExecutionProvider(SharedConfig.GetTestModelV26(ModelType.ObjectDetection))
});
```

**2. Capture frames.** A background task reads frames from the webcam with [OpenCvSharp](https://github.com/shimat/opencvsharp),
converts them to BGRA and copies them into an `SKBitmap`.

```csharp
capture.Read(mat);
Cv2.CvtColor(mat, bgraMat, ColorConversionCodes.BGR2BGRA);
_currentFrame.SetPixels(bgraMat.Data);
```

**3. Detect, filter, track and draw.** While detection is on, each frame goes through the model. The results are filtered and tracked
if those options are checked, then drawn on the frame.

```csharp
var results = _yolo.RunObjectDetection(_currentFrame, _confidenceThreshold, iou: 0.5);

if (_isFilteringEnabled)
    results = results.FilterLabels(_filteredLabels);

if (_isTrackingEnabled)
    results.Track(_sortTracker);

_currentFrame.Draw(results);
```

**4. Show the frame.** The UI thread repaints the `SKElement` in the window with the current frame and updates the timing text.

Closing the window stops the capture loop and disposes the model.

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
| `enableCudaGraph` | `false` | CUDA Graph replay, which removes most of the per-frame GPU launch overhead and lowers latency; see the [CUDA execution provider README](../../YoloDotNet.ExecutionProvider.Cuda/README.md). Useful for live video like this. |

To run the demo on the CPU, or with another provider, see [Execution providers](../README.md#-execution-providers).

### Model settings (`YoloOptions`)

| Property | Default | Description |
|----------|---------|-------------|
| `ImageResize` | `Proportional` | `Proportional` keeps the aspect ratio and pads the image; `Stretched` resizes it to the model's input size. Use the mode your model was trained with. |
| `SamplingOptions` | Bilinear | How the image is resized. The default matches how YOLO models are trained. |

### Detection (`RunObjectDetection`)

| Parameter | Default | This demo | Description |
|-----------|---------|-----------|-------------|
| `confidence` | `0.2` | Slider (0.80 at start) | Minimum confidence for an object to be returned. |
| `iou` | `0.7` | `0.5` | Overlap threshold for removing duplicate boxes (non-maximum suppression). Lower values remove more overlapping boxes. |
| `roi` | `null` | – | Optional region of interest (`SKRectI`); only objects inside it are detected. |

📖 For guidance on choosing these values, see the [Accuracy & Configuration Guide](../../AccuracyAndConfiguration.md).

### Filtering (`FilterLabels`)

`results.FilterLabels(labels)` keeps only the results whose label is in the set. The demo uses:

```csharp
private static readonly HashSet<string> _filteredLabels = ["person", "cat", "dog"];
```

### Tracking (`SortTracker`)

`results.Track(tracker)` gives each object a stable id across frames and records its path. Use one tracker per video stream.
The demo creates it with `new SortTracker(maxAge: 5)`.

| Parameter | Default | This demo | Description |
|-----------|---------|-----------|-------------|
| `costThreshold` | `0.5` | – | Maximum matching cost for assigning a detection to a track. Lower values match more strictly. |
| `maxAge` | `3` | `5` | Number of frames an unmatched track is kept. Higher values survive longer occlusions. |
| `tailLength` | `30` | – | Maximum number of points in each track's tail. |
| `highConfidenceThreshold` | `0` | – | When above 0, enables ByteTrack-style two-stage matching: detections below it only keep existing tracks alive. Run inference with a lower confidence to benefit from it. |
| `minHits` | `3` | – | Number of matched frames before a new track gets an id. |

There is no one-size-fits-all setting; expect to tune these for your scene.

### Webcam

The capture settings are constants at the top of `MainWindow.xaml.cs`:

| Constant | Value | Description |
|----------|-------|-------------|
| `CAMERA_INDEX` | `0` | The webcam to use: `0` is the default camera, `1` the next, and so on. |
| `WEBCAM_WIDTH` × `WEBCAM_HEIGHT` | `1280` × `720` | Capture resolution. If you change it, change the `SKElement` size in `MainWindow.xaml` to match. |
| `FPS` | `30` | Requested capture frame rate. |

The demo also asks the camera for MJPEG frames, which are much cheaper on the CPU than uncompressed frames, if the camera supports them.
The resolution and frame rate must be a mode your camera supports.

### Drawing

The demo draws the results with the default `DetectionDrawingOptions`: boxes, labels and confidence scores, plus tails for tracked objects.
To change how they look, pass your own options to `Draw`; see [Drawing in the Object Detection Demo](../ObjectDetectionDemo/README.md#drawing-detectiondrawingoptions) for every option.

```csharp
_currentFrame.Draw(results, new DetectionDrawingOptions
{
    DrawLabels = false,
    TailThickness = 6
});
```

## 📚 See also

- [Demo README](../README.md): setup and the steps all demos share
- [VideoStreamDemo](../VideoStreamDemo): object detection and tracking on video files, livestreams and webcams, without WPF
- [Object Detection Demo](../ObjectDetectionDemo/README.md): object detection on a single image, with every drawing option
