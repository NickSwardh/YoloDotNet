# Changelog

All notable changes to **YoloDotNet** will be documented in this file.

## [4.3] — The Fastest YoloDotNet Yet, Accurate Segmentation, Trim-Safe Loading

> **🔥 The fastest YoloDotNet yet, optimized for speed.** Faster from preprocessing to drawing, with less memory; the speed optimizations don't change the detections.
> Compared with v4.2 (yolo26s, default settings, RTX 5070 Ti; gains vary by hardware and model):
> - **Video, TensorRT FP16:** 290 → **714 fps** with `StartVideoProcessing<T>` and CUDA Graphs (**2.5× faster**); 455 fps without graphs
> - **Video, CUDA:** 157 → **268 fps** with CUDA Graphs (**1.7× faster**); 197 fps without graphs
> - **Detection, TensorRT FP16:** 2.47 → **1.40 ms** per image with CUDA Graphs (**1.8× faster**); 2.11 ms without graphs
> - **Detection, CUDA:** 5.27 → **3.51 ms** with CUDA Graphs (**1.5× faster**); 4.88 ms without graphs
> - **Segmentation, CUDA:** 7.55 → **4.91 ms** with CUDA Graphs (**1.5× faster**); 6.48 ms without graphs
> - **Drawing segmentation results** (masks, boxes and labels): 1.68 → **0.71 ms** (**2.4× faster**), while drawing more masks (33 instead of 28)
> - **Lower memory:** YOLOv26 detection allocates 40% less per call
> - **Smaller apps:** trimming support makes a self-contained app less than half the size (97 → 44 MB)
>
> The numbers in the sections below were measured per change, against the code just before that change.

### 🚨 Breaking Changes
- **Execution providers:** this version requires the updated execution provider packages (**v1.2**). Older providers stop at model load with a message saying so.
  Custom execution providers build the model information with `ParseOnnxData.CreateOnnxModel(...)` instead of `ParseOnnx(object)`.
- **Default resize sampling** changed from nearest-neighbour to **bilinear** (`SKFilterMode.Linear`), matching how YOLO models are trained and validated.
  Detections are more accurate but can differ slightly from earlier versions. Set `YoloOptions.SamplingOptions = new SKSamplingOptions(SKFilterMode.Nearest)` for the old behaviour.
- **Pose drawing:** `PoseDrawingOptions.KeyPointMarkers`, `KeyPointMarker` and `KeyPointConnection` are **obsolete and have no effect**.
  Existing code still compiles (with a warning) but no longer draws keypoint colors and connections. Use `PoseDrawingOptions.PoseKeyPoints` instead (see *Added*).
- **Tracking:** `SortTracker` now confirms a new track only after `minHits` matched frames (default **3**). Until then the track has no Id (`null`).
  Use `minHits: 1` to assign Ids immediately, as before.
- Removed `ImageExtension.CreateGrayscaleToColorFilter`, `ObjectResult.BoundingBoxUnscaled` and `Utils.Avx2LinearResizer` (no longer used).

### 🚀 Improved — Faster, Leaner & More Accurate
- **Segmentation masks** are traced as outlines instead of evaluated per pixel:
  - Masks agree much more closely with Ultralytics (mean mask IoU 0.81 → 0.97); the previous shift and shrink of masks is gone.
  - Post-processing about **2.5× faster**, mask drawing about **6× faster**, contour drawing about **4× faster**, and roughly **4× less memory** per mask.
  - Masks and contours are drawn anti-aliased with sub-pixel edges.
- **Faster preprocessing:** a new AVX2 bilinear resize for 32-bit images resizes and normalizes in a single pass (FP32 and FP16 models),
  and large images are resized on up to 4 threads. SIMD RGB normalization and an FP16 lookup table speed up the remaining paths.
  About **10% faster** per detection on a GPU, with identical results.
- **Parallel inference:** running inference on the same `Yolo` instance from multiple threads is now thread-safe.
- **Trimmed apps:** model loading no longer uses reflection or `dynamic`, and video metadata is read with source-generated JSON,
  so YoloDotNet works in apps published with trimming (`PublishTrimmed`). Previously every model failed to load there.
  A self-contained test app went from 97 MB to 44 MB.
- **Faster segmentation and pose drawing** on CPUs with AVX2: masks and keypoints are filled by YoloDotNet's own single-threaded code.
  Masks are drawn **up to 43% faster**, keypoints with their skeleton **up to 19% faster**; `BitPackedPixelMask` is built up to 62% faster
  and `GetPixelMask` up to 32% faster. Lines, outlines, boxes and labels are drawn by SkiaSharp as before. Without AVX2 (older CPUs, ARM) drawing is unchanged.
- **Bounding boxes and keypoints** (all tasks) are rounded instead of truncated and the letterbox matches Ultralytics exactly, removing a systematic ~0.5 px offset.
  Results can move by up to about 1 px.
- **TensorRT** sessions also register CUDA, so operations TensorRT cannot run stay on the GPU instead of falling back to the CPU.
- **YOLOv26 detection** allocates about 40% less memory per call.
- **Pose drawing:** connection lines stop at the keypoint edges, so they no longer show through semi-transparent keypoints.
- Faster best-label search in YOLOv8 detection and the versions sharing its architecture.

### ✨ Added
- **FP16 half models:** models with FP16 input and FP32 outputs (for example exported with `half=True`), or one FP32 and one FP16 output
  (segmentation), are now supported on every execution provider, alongside FP32 and FP16 full models. Each output is read by its own type.
  Before, these models failed with "Requested type: Float16 does not match the actual type: System.Single".
- **Faster video processing:** `StartVideoProcessing<T>(onResult, confidence, iou, ...)` runs the loaded model itself and detects upcoming frames
  while your handler processes the current one, still one frame at a time and in frame order (tracking works as before).
  Video files are processed **16–86% faster**, depending on task and execution provider (detection 45–86%). `OnVideoFrameReceived` with `StartVideoProcessing()` still works.
- **CUDA Graphs (opt-in), for CUDA and TensorRT:** `enableCudaGraph: true` on `CudaExecutionProvider` replays the recorded GPU work instead of
  launching it step by step, with identical results: about **14–33% faster** per inference on CUDA and **23–38% faster** on TensorRT
  (`new CudaExecutionProvider(model, trtConfig: tensorRt, enableCudaGraph: true)`). Requires a model that runs entirely on the GPU.
- **Pose keypoints from JSON:** define keypoint names, colors and connections in a JSON file and load it once with `PoseKeyPoints.Load(path)`
  (or `Load(Stream)` / `Parse(json)`), then set `PoseDrawingOptions.PoseKeyPoints`. Works with any pose model.
  - Keypoints and connections are referenced by name; colors can come from a named palette.
  - Invalid files fail with an error pointing to the exact JSON path, e.g. `connections[2].to`.
  - `ValidateKeyPointCount(int)` to check the file against your model at startup, and `ToJson()` to export a template.
  - A COCO-17 file for the Ultralytics pose models ships with the PoseEstimation demo (`pose-keypoints.json`).
- **Segmentation outlines:** `Segmentation.Contours` (`SKPoint[][]`) — the mask as sub-pixel polygons in image coordinates, including holes and separate parts.
- `Segmentation.GetPixelMask(antiAliased)` and `GetPixelMask(imageWidth, imageHeight, antiAliased)` return the mask as a Gray8 bitmap, optionally anti-aliased.
- **Tracking:** new `SortTracker` options `highConfidenceThreshold` (ByteTrack-style two-stage matching that keeps partially occluded objects tracked) and `minHits`.
  `TrackedObject` exposes `Id`, `Hits` and `GetPredictedBox()`.
- **Clearer errors:** a `null` image (for example a file that failed to load) and a missing video file are reported as such, instead of
  "Unsupported image type" and "invalid video stream". A model exported with dynamic input sizes (`dynamic=True`) is reported as not supported,
  with how to export it, instead of "All shape dimensions must be positive".

### 🔨 Bugs Crushed
- **YOLOv26 segmentation and pose:** concurrent calls on one `Yolo` instance could mix up masks or keypoints between objects.
- **YOLOv10, Stretched mode:** boxes were decoded as centre/size instead of corners, placing them far off.
- **YOLOv8/v11/v12 pose, Stretched mode:** keypoints are now placed accurately instead of being clamped inside the bounding box.
- **Tracking:** corrected LAPJV assignment and improved `SortTracker` Id stability.
- **Video:** `Dispose` now releases the FFmpeg encode process; previously it released the decode process twice and never the encoder.
- **Pose drawing:** a keypoint-count mismatch between the model and the drawing options no longer throws while drawing.
- **OpenVINO:** with `DeviceType` set to `"GPU"` or `"NPU"`, leaving `CachePath` unset threw a `NullReferenceException`.
  `CachePath` is now optional; without it, OpenVINO compiles the model on every start.
- **Saving images:** an image saved with `Save(...)` right before the program exited could be lost, and images saved before any `Yolo`
  instance was created were never written. Every saved image is now written before the process exits. Saving over a larger existing file
  no longer leaves the old file's trailing bytes, and a failed write no longer stops later images from being saved.

### 🔄 Changed
- `GetContourPoints(maxPoints)` returns the main outline with sub-pixel points and simplifies long outlines (Ramer–Douglas–Peucker) instead of skipping points.
  Segmentation polygons in YOLO-format and JSON exports change accordingly.
- **On CPUs with AVX2, mask and keypoint edges use the exact covered area:** edge pixels of drawn masks and keypoints, and of
  `GetPixelMask(antiAliased: true)`, can differ from earlier versions by a few colour levels; inside masks by at most 2.
  A smooth pixel mask now matches the drawn mask exactly. Sharp masks (`GetPixelMask()`, `BitPackedPixelMask`) only differ
  for pixels whose centre lies on the outline (within 0.03 px).
- With an ROI, segmentation `Contours` are returned in full-image coordinates together with the box.
- `Segmentation.BitPackedPixelMask` keeps its type and layout, but is now built from `Contours` the first time it is read.

### ⚠️ Deprecated
- `ImageExtension.UnpackToBitmap` and `UnpackPixelMaskToByteArray` — use `Segmentation.GetPixelMask()` (returns Gray8: 255 = mask, 0 = background).

## [4.2] — Usability, ROI & Segmentation Enhancements

### ✨ Added
- Option to define a **Region of Interest (ROI)** to run inference on specific areas of an image or video stream.
  - Useful for surveillance, monitoring zones, and performance optimization.
- Option to **draw edges** on segmented objects.
- Helper methods: `ToJson()` and `SaveJson()` for exporting inference results as JSON.
- Helper methods: `ToYoloFormat()` and `SaveYoloFormat()` for saving inference results as YOLO-annotated data.
- Helper method: `GetContourPoints()` for extracting ordered contour points from segmented objects.

### 🔄 Changed
- Updated **YOLOv26 inference execution** to align with other tasks, improving consistency and overall execution efficiency.

## [4.1] — Expanded Model Support & MIT Relicense

### ✨ Added
- Support for the **YOLOv26** model suite.
- Support for **RT-DETR** models.
- New **DirectML execution provider** for GPU acceleration on Windows.
- Improved video inference pipeline with better frame-to-frame stability.

### 🚀 Improved
- Reduced allocation pressure across all vision tasks.
- Lower per-frame latency for long-running inference workloads.
- General performance improvements across CPU and GPU execution providers.

### 🔄 Changed
- Project relicensed to **MIT** to maximize freedom for commercial and proprietary use.
  - No copyleft requirements
  - No network-use clauses
  - Model licensing remains the responsibility of the user

## [4.0] — Modular Execution Providers

### 🚨 Breaking Changes
- Execution providers are no longer bundled with the core package.
- Consumers **must reference exactly one execution provider NuGet package**.
- Provider setup code must be updated when upgrading from v3.x.

### ✨ Added
- Fully modular execution provider architecture.
- New execution providers:
  - **CPU** (baseline, cross-platform)
  - **CUDA / TensorRT** (NVIDIA GPUs, optional TensorRT acceleration)
  - **OpenVINO** (Intel CPUs and iGPUs on Windows/Linux)
  - **CoreML** (Apple Silicon on macOS)
- Support for **grayscale-only ONNX models**.
- Improved GPU execution behavior and predictability.
- Clear separation of native ONNX Runtime dependencies per provider.

### 🔧 Changed
- Core package is now execution-provider agnostic.
- Dependency graph simplified for predictable deployment.
- CUDA provider exposes clearer and more explicit GPU behavior.

### ⬆️ Updated
- **SkiaSharp** → 3.119.1
- **ONNX Runtime (CUDA)** → 1.23.2

---

## [Pre-4.0]

Changes prior to version **4.0** were released before a formal changelog was introduced.

For historical context, refer to:
- [NuGet package history](https://www.nuget.org/packages/YoloDotNet/#versions-body-tab)
