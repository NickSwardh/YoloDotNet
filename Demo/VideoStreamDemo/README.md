# <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/994287a9-556c-495f-8acf-1acae8d64ac0" height=24> Video Stream Demo
**Detect and track objects in videos, livestreams and webcams**

This demo runs a YOLO26 object detection model on every frame of a video, keeps only people, cats and dogs,
tracks them across frames with the SORT tracker, and saves a new video with bounding boxes, labels and movement tails drawn on it.

## ▶️ Run the demo

Before you start, export the models and check the GPU requirements; see the [Demo README](../README.md#-quick-start).

This demo also needs:

- **FFmpeg and FFprobe** on your `PATH` ([download](https://ffmpeg.org/download.html)). Check that `ffmpeg -version` works in a terminal.
- **An NVIDIA GPU** for both inference (CUDA) and encoding (NVENC). FFmpeg must be built with NVENC support; otherwise, switch to the `LibX264` encoder (see [Video options](#video-options-videooptions)).

From the repository root:

```bash
dotnet run --project Demo/VideoStreamDemo
```

The demo lists the video input devices it finds, prints the input and output video size, shows the progress,
saves the result to `YoloDotNet_Results/video_output.mp4` on your desktop and opens that folder.

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

**2. Set up the video.** `VideoOptions` sets the input, where to save the result and how to encode it.
The output is scaled to 720 pixels wide, keeping the aspect ratio, and saved as one file.

```csharp
yolo.InitializeVideo(new VideoOptions
{
    VideoInput = SharedConfig.GetTestVideo(VideoType.Default),
    VideoOutput = Path.Combine(outputFolder, "video_output.mp4"),
    VideoEncoder = VideoEncoder.H264Nvenc,
    Width = 720,
    Height = -2,
    VideoChunkDuration = 0
});
```

**3. Create a tracker.** The SORT tracker gives each object an id that stays the same across frames, and records its path for the tail.

```csharp
var tracker = new SortTracker(costThreshold: 0.7f, maxAge: 30, tailLength: 60, highConfidenceThreshold: 0.25f);
```

**4. Process the video.** `StartVideoProcessing<ObjectDetection>` runs the model on every frame and calls your handler with the frame,
its results and its index. The handler filters the results, tracks them and draws them on the frame; drawing on the frame is kept in the output video.

```csharp
yolo.StartVideoProcessing<ObjectDetection>((frame, results, frameIndex) =>
{
    results = results
        .FilterLabels(["person", "cat", "dog"])
        .Track(tracker);

    frame.Draw(results);
},
confidence: 0.1, iou: 0.5);
```

The handler runs one frame at a time and in frame order, so tracking and other per-frame state work as expected.
Meanwhile, YoloDotNet already detects the next frames, which keeps the GPU busy and processes video files considerably faster.
The type argument must match the model's task: `ObjectDetection`, `Segmentation`, `PoseEstimation`, `OBBDetection` or `Classification`.

`SharedConfig` supplies the model and the test video (`test/assets/Media/walking.mp4`), so the demo runs without any paths to set.
In your own code, pass the path to your model and video source instead.

## 🎥 Video sources

`VideoInput` accepts three kinds of source:

| Source | Example |
|--------|---------|
| Video file | `C:\videos\test.mp4` |
| Livestream URL (RTMP, HTTP, …) | `rtmp://your.server/stream` |
| Video capture device (webcam) | `device=<DeviceName>:<Width>:<Height>:<FPS>` |

Device examples:

- Windows: `device=Logitech BRIO:1920:1080:30`
- Linux: `device=/dev/video0:1280:720:30`

> [!NOTE]
> The width, height and frame rate must match a capture mode your device supports.
> `Yolo.GetVideoDevices()` lists the device names on your system (the demo prints them at startup), but not their resolutions or frame rates;
> check your device's specifications for those.

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
| `enableCudaGraph` | `false` | CUDA Graph replay; see the tip below. |

> [!TIP]
> `enableCudaGraph: true` removes most of the per-frame GPU launch overhead: about 25–40% faster per frame on YOLO26 models.
> It also works together with `trtConfig`. It requires a model that runs entirely on the GPU;
> see the [CUDA execution provider README](../../YoloDotNet.ExecutionProvider.Cuda/README.md).

To run the demo on the CPU, or with another provider, see [Execution providers](../README.md#-execution-providers).

### Model settings (`YoloOptions`)

| Property | Default | Description |
|----------|---------|-------------|
| `ImageResize` | `Proportional` | `Proportional` keeps the aspect ratio and pads the image; `Stretched` resizes it to the model's input size. Use the mode your model was trained with. |
| `SamplingOptions` | Bilinear | How the image is resized. The default matches how YOLO models are trained. |

### Video options (`VideoOptions`)

| Property | Default | This demo | Description |
|----------|---------|-----------|-------------|
| `VideoInput` | – | Test video | The video source; see [Video sources](#-video-sources). |
| `VideoOutput` | `null` | `video_output.mp4` | Where to save the processed video. Leave unset to not save it. |
| `VideoEncoder` | `LibX264` | `H264Nvenc` | The encoder for the output video. See below. |
| `Width` | `0` | `720` | Output width in pixels. `0` keeps the source width; `-2` calculates it from `Height`, keeping the aspect ratio. |
| `Height` | `0` | `-2` | Output height in pixels. `0` keeps the source height; `-2` calculates it from `Width`, keeping the aspect ratio. Only one of `Width` and `Height` can be `-2`. |
| `FrameRate` | `FrameRate.AUTO` | – | Output frame rate. `AUTO` matches the input. |
| `CompressionQuality` | `30` | – | Output compression, from 0 to 51. Lower values give better quality and larger files; 20–35 is a good range. |
| `VideoChunkDuration` | `600` | `0` | Splits the output into chunks of this many seconds. `0` saves one file. |
| `FrameInterval` | `0` | – | Processes only every Nth frame, for example `30` for surveillance. `0` processes every frame. |
| `StartTimeSeconds` | `0` | – | Where to start in a video file. Not used for livestreams. |
| `DurationSeconds` | `0` | – | How many seconds of a video file to process. `0` processes to the end. Not used for livestreams. |

**Encoders:** the encoder must be supported by your hardware, by your FFmpeg build and by the output container.

| Codec | CPU | NVIDIA | Intel | AMD | Linux (VA-API) | Apple |
|-------|-----|--------|-------|-----|----------------|-------|
| H.264 | `LibX264` | `H264Nvenc` | `H264Qsv` | `H264Amf` | `H264Vaapi` | `H264VideoToolbox` |
| HEVC | `LibX265` | `HevcNvenc` | `HevcQsv` | `HevcAmf` | `HevcVaapi` | `HevcVideoToolbox` |
| AV1 | `LibAomAv1` | `Av1Nvenc` | `Av1Qsv` | – | `Av1Vaapi` | `Av1VideoToolbox` |
| ProRes | – | – | – | – | – | `ProResVideoToolbox` |

### Processing (`StartVideoProcessing<T>`)

| Parameter | Default | This demo | Description |
|-----------|---------|-----------|-------------|
| `onResult` | – | – | Your handler, called with each frame, its results and its index. |
| `confidence` | `0.2` | `0.1` | Minimum confidence for a result. The demo uses a value below the tracker's `highConfidenceThreshold`, so weak detections can keep tracks alive. |
| `iou` | `0.7` | `0.5` | Overlap threshold for removing duplicate boxes (non-maximum suppression). |
| `roi` | `null` | – | Optional region of interest (`SKRectI`) in each frame. |
| `pixelConfidence` | `0.65` | – | Segmentation only: minimum confidence for a pixel to belong to a mask. |
| `classes` | `1` | – | Classification only: how many of the most likely classes to return. |
| `framesInFlight` | `2` | – | How many frames are detected at once. |

`yolo.OnVideoEnd` is called once when processing finishes; the demo uses it to report whether the whole video was processed.
`yolo.StopVideoProcessing()` stops processing early, for example for a livestream.

**Running detection yourself:** when each frame needs its own detection, such as a different region per frame, several models,
or skipping frames, handle `OnVideoFrameReceived` instead. Each frame is then detected and handled one after the other, without the overlap:

```csharp
yolo.OnVideoFrameReceived = (frame, frameIndex) =>
{
    var results = yolo.RunObjectDetection(frame, confidence: 0.1, iou: 0.5)
        .FilterLabels(["person", "cat", "dog"])
        .Track(tracker);

    frame.Draw(results);
};

yolo.StartVideoProcessing();
```

### Tracking (`SortTracker`)

| Parameter | Default | This demo | Description |
|-----------|---------|-----------|-------------|
| `costThreshold` | `0.5` | `0.7` | Maximum matching cost for assigning a detection to a track. Lower values match more strictly. The cost is `(1 - IoU)` plus the center distance relative to the object's size. |
| `maxAge` | `3` | `30` | How many frames an unmatched track is kept. Higher values survive longer occlusions. |
| `tailLength` | `30` | `60` | How many points of each object's path are kept for the tail. |
| `highConfidenceThreshold` | `0` | `0.25` | Above `0`, enables ByteTrack-style matching: detections at or above this confidence can start tracks, weaker ones only keep existing tracks alive (for example, partially hidden people). |
| `minHits` | `3` | – | How many matched frames a new track needs before it gets an id. `1` assigns ids immediately. |

There is no setting that fits every video; expect to tune these for your use case.

### Filtering

`FilterLabels` keeps only the results with the given class names:

```csharp
results = results.FilterLabels(["person", "cat", "dog"]);
```

### Drawing

`frame.Draw(results)` uses the default `DetectionDrawingOptions`; see the [ObjectDetectionDemo README](../ObjectDetectionDemo/README.md#drawing-detectiondrawingoptions) for all properties.
The tail properties apply to tracked objects:

| Property | Default | Description |
|----------|---------|-------------|
| `DrawTrackedTail` | `true` | Draw a tail along each tracked object's path. |
| `TailThickness` | `4` | Tail thickness, in pixels. |
| `TailPaintColorStart` | `#FF69B4` (pink) | Tail color at the object. |
| `TailPaintColorEnd` | `TailPaintColorStart`, fully transparent | Tail color at the end, so the tail fades out. |

### Saving frames as images

To save individual frames, for example for debugging or to build an image dataset, save the frame in your handler:

```csharp
frame.Save(Path.Combine(outputFolder, $"frame_{frameIndex}.jpg"), SKEncodedImageFormat.Jpeg, 80);
```

## 💡 Use cases

- **Surveillance and security:** track people and vehicles, watch entrances or restricted zones, record annotated video for later review.
- **Industrial automation:** detect people in hazardous machine zones, monitor conveyor belts, count packages.
- **Traffic and smart cities:** follow pedestrian flow at crossings, analyze vehicle movement, spot vehicles parked in no-parking zones.
- **Live streaming:** broadcast livestreams with real-time detection overlays, for example sports or wildlife cameras.
- **Research and prototyping:** benchmark models on video, collect results for further analysis, compare confidence and IoU settings.

## 📚 See also

- [Demo README](../README.md): setup and the steps all demos share
- [ObjectDetectionDemo](../ObjectDetectionDemo/README.md): object detection on a single image, with all drawing options
- [Accuracy & Configuration Guide](../../AccuracyAndConfiguration.md): resize mode, sampling and thresholds
- [CUDA execution provider README](../../YoloDotNet.ExecutionProvider.Cuda/README.md): CUDA, TensorRT and CUDA Graphs
