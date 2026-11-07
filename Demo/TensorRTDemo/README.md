# <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/994287a9-556c-495f-8acf-1acae8d64ac0" height=24> TensorRT Demo
**GPU-accelerated object detection with NVIDIA TensorRT**

TensorRT is NVIDIA's inference optimizer and runtime. It builds an engine tuned for your GPU and model,
which gives lower latency and higher throughput than plain CUDA inference.

This demo runs YOLO26 object detection with TensorRT in FP16 precision and CUDA Graph replay, prints what it found,
and saves the image with bounding boxes, labels and confidence scores drawn on it.

## ▶️ Run the demo

Before you start:

- Export the models; see the [Demo README](../README.md#-quick-start).
- Install CUDA, cuDNN and TensorRT; see [Installing CUDA and TensorRT](#-installing-cuda-and-tensorrt) below.

From the repository root:

```bash
dotnet run --project Demo/TensorRTDemo
```

The first run builds the TensorRT engine, which can take several seconds to a few minutes; see [Engine cache](#-engine-cache).
The demo then prints each detected object with its confidence, saves the result to `YoloDotNet_Results/TensorRT.jpg` on your desktop and opens that folder.

## 🧩 How it works

[`Program.cs`](./Program.cs) creates the engine cache folder (`YoloDotNet_Results/TensorRT_Engine_Cache` on your desktop) and then runs four steps:

**1. Load the model with TensorRT.** The demo loads `yolo26s.onnx` from the repository's `test/assets/Models` folder.
If the models are missing, run `python tools/download_models.py` from the repository root to download and export them into that folder.
TensorRT is enabled by passing a `TensorRt` configuration to the CUDA execution provider:

```csharp
using var yolo = new Yolo(new YoloOptions
{
    ExecutionProvider = new CudaExecutionProvider(
        model: SharedConfig.GetTestModelV26(ModelType.ObjectDetection),
        trtConfig: new TensorRt
        {
            Precision = TrtPrecision.FP16,
            EngineCachePath = engineCacheFolder,
            Int8CalibrationCacheFile = ...   // only used with TrtPrecision.INT8
        },
        enableCudaGraph: true)
});
```

**2. Run inference.** `RunObjectDetection` returns a list of `ObjectDetection` results, each with a label, a confidence and a bounding box.

```csharp
using var image = SKBitmap.Decode(SharedConfig.GetTestImage(ImageType.ObjectDetection));
var results = yolo.RunObjectDetection(image, confidence: 0.15);
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

| Parameter | Default | This demo | Description |
|-----------|---------|-----------|-------------|
| `model` | – | `yolo26s.onnx` | Path to the ONNX model, or the model as a `byte[]`. |
| `gpuId` | `0` | `0` | The GPU to run on. |
| `trtConfig` | `null` | see below | TensorRT settings. Without them, the provider runs plain CUDA. |
| `enableCudaGraph` | `false` | `true` | Records the GPU work on the first inference and replays it in one launch afterwards: faster per inference, identical results. The whole model must run on TensorRT; models that don't qualify fail when the provider is created. See the [CUDA execution provider README](../../YoloDotNet.ExecutionProvider.Cuda/README.md). |

### TensorRT settings (`TensorRt`)

| Property | Default | This demo | Description |
|----------|---------|-----------|-------------|
| `Precision` | FP32 | `FP16` | `FP32`: full precision, highest accuracy. `FP16`: half precision, faster with minimal accuracy loss on supported GPUs. `INT8`: fastest, but needs a calibration cache; see [INT8 calibration cache](#-int8-calibration-cache). |
| `BuilderOptimizationLevel` | `3` | `3` | How much effort TensorRT spends optimizing when it builds the engine, from 0 to 5. Higher levels build slower but can run faster; levels below 3 build faster but don't guarantee good performance. |
| `EngineCachePath` | Not set | Engine cache folder | Folder where TensorRT stores and loads its engine. The folder must exist. If not set, the engine cache is disabled and the engine is rebuilt on every start. |
| `EngineCachePrefix` | `YoloDotNet` | `YoloDotNet` | File name prefix for the engine cache files. Use different prefixes to keep engines for several models or configurations in the same folder. |
| `Int8CalibrationCacheFile` | Not set | `test/assets/Cache/yolo26s.cache` | Path to the INT8 calibration cache for the model. Required with `INT8` (the file must exist); ignored otherwise. The repository doesn't include `yolo26s.cache`; to use INT8, generate it first; see [INT8 calibration cache](#-int8-calibration-cache). |

### Detection (`RunObjectDetection`)

| Parameter | Default | This demo | Description |
|-----------|---------|-----------|-------------|
| `confidence` | `0.2` | `0.15` | Minimum confidence for an object to be returned. |
| `iou` | `0.7` | `0.7` | Overlap threshold for removing duplicate boxes (non-maximum suppression). |
| `roi` | `null` | – | Optional region of interest (`SKRectI`); only objects inside it are detected. |

### Drawing

The demo draws with the default `DetectionDrawingOptions`. For all drawing options and their defaults,
see [Drawing](../ObjectDetectionDemo/README.md#drawing-detectiondrawingoptions) in the ObjectDetectionDemo README.

## 🧰 Installing CUDA and TensorRT

You need:

1. An NVIDIA GPU with a supported CUDA compute capability.
2. CUDA 12 and cuDNN 9.
3. The TensorRT 10 runtime libraries.

### CUDA

If you already have CUDA and cuDNN installed, skip to [TensorRT](#tensorrt).

1. Download and install from NVIDIA:
    - [CUDA 12.x](https://developer.nvidia.com/cuda-downloads)
    - [cuDNN 9.x](https://developer.nvidia.com/cudnn-downloads)
2. Locate the folder with the cuDNN DLL files, typically:

    ```
    C:\Program Files\NVIDIA\CUDNN\v9.x\bin\v12.x
    ```

    *(Replace v9.x and v12.x with your installed versions.)*
3. Copy the full path of that folder from the File Explorer address bar.
4. Add it to the Windows `PATH`:
    - Search for **Edit the system environment variables** in the Start menu and open it.
    - Click **Environment Variables**.
    - Under **System variables**, select `Path` and click **Edit**.
    - Click **New** and paste the cuDNN folder path.
    - Click **OK** to close all dialogs.
5. Restart any open programs, or reboot, so Windows picks up the new `PATH`.

### TensorRT

1. Go to [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) and click **Download Now** (you may need to sign in with an NVIDIA account).
2. Select **TensorRT 10**, accept the terms and choose the latest TensorRT 10 release.
3. Download the Windows **ZIP package** for **CUDA 12.x**.
4. Unzip it to a folder on your system.
5. Copy the full path of the `lib` folder inside the extracted TensorRT folder.
6. Add it to the Windows `PATH`, the same way as for cuDNN (CUDA step 4).
7. Restart any open programs, or reboot, so the new `PATH` takes effect.

## 📁 Engine cache

### First run: building the engine

When there is no engine that matches the model and settings, TensorRT builds one. This happens on the first run,
after changing the model or a setting (such as precision or optimization level), and after deleting the cache.
Building includes layer fusion, kernel auto-tuning, precision calibration and saving the engine to disk.

⏳ It takes from several seconds to a few minutes, depending on the GPU, the size of the model, the precision and `BuilderOptimizationLevel`.
Later runs load the saved engine from `EngineCachePath` and start quickly.

### Managing cache files

Neither YoloDotNet nor TensorRT deletes engine cache files. Each change of model or setting creates a new engine file,
and the old ones stay. Remove outdated files yourself to save disk space and avoid confusion.
Make sure `EngineCachePath` points to a writable folder that persists between runs, and keep useful engines for reuse in production.

### 🚫 Don't share engine files between machines

An engine is built for one specific setup: the system architecture, the installed CUDA and TensorRT versions,
the exact GPU model, and the `TensorRt` settings. It doesn't work elsewhere.

- Don't copy engine files to other machines unless the hardware and software are identical.
- Let TensorRT build the engine on each target machine; it is then reused there.
- Use `EngineCachePrefix` to keep engines for several models or configurations side by side.

## 🔢 INT8 calibration cache

INT8 precision needs a calibration cache. It holds the value ranges of the model's tensors,
so TensorRT can quantize the layers that tolerate it without losing accuracy.

> [!NOTE]
> INT8 runs in mixed precision: TensorRT uses INT8 where it can and falls back to FP16 or FP32 for layers
> that can't be quantized (unsupported operations, numerical stability or dynamic range).

> [!WARNING]
> YoloDotNet doesn't support INT8 for segmentation models. Using INT8 with a segmentation model throws a `YoloDotNetModelException`.

### Generating the calibration cache

1. Install [Ultralytics](https://docs.ultralytics.com/quickstart/).
2. Export the model to a TensorRT engine with INT8 calibration, using a dataset that represents your data:

    ```bash
    yolo export model=your_model.pt format=engine int8=true simplify=true data=your_dataset.yaml opset=17
    ```

    Replace `your_model.pt` with your PyTorch YOLO model and `your_dataset.yaml` with your dataset YAML file.
3. The export creates:
    - a `.engine` file: the TensorRT engine (not needed by YoloDotNet)
    - a `.cache` file: the calibration cache
4. Point YoloDotNet to the cache:

    ```csharp
    Int8CalibrationCacheFile = @"C:\path\to\your_model.cache"
    ```

The calibration cache can be reused and shipped with your application.

> [!IMPORTANT]
> TensorRT doesn't overwrite an existing calibration cache. To regenerate it, delete the old `.cache` file before exporting again.

## 📚 See also

- [Demo README](../README.md): setup and the steps all demos share
- [CUDA execution provider README](../../YoloDotNet.ExecutionProvider.Cuda/README.md): CUDA requirements and CUDA Graph replay
- [ObjectDetectionDemo](../ObjectDetectionDemo/README.md): the same detection without TensorRT, and all drawing options
