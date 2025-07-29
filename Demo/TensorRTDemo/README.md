# YoloDotNet with TensorRT Execution Provider

TensorRT is NVIDIA’s high-performance deep learning inference optimizer and runtime library. It accelerates neural network inference by optimizing models and efficiently running them on NVIDIA GPUs. By leveraging TensorRT, applications can achieve significant speed-ups, reduced latency, and lower power consumption compared to standard GPU or CPU inference.

In YoloDotNet, TensorRT can be used as an execution provider to perform GPU-accelerated inference of YOLO ONNX models. This is particularly beneficial for real-time or high-throughput object detection applications where inference speed is critical.

# Installation Steps

To use TensorRT with YoloDotNet, you need:

1. A compatible NVIDIA GPU with the appropriate CUDA Compute Capability.
2. `CUDA Toolkit` installed on your system.
3. `TensorRT` runtime libraries installed.

## CUDA - Installation Steps

If you already have CUDA installed, skip this step and jump to the [TensorRT Installation](#tensorrt---installation-steps) below.

1. Download and install the following from NVIDIA’s official sites:
    - [CUDA v12.x](https://developer.nvidia.com/cuda-downloads)
    - [cuDNN v9.x](https://developer.nvidia.com/cudnn-downloads)

2. Locate the folder containing the cuDNN DLL files, typically:
    \
    ```
    C:\Program Files\NVIDIA\CUDNN\v9.x\bin\v12.x
    ```
    *(Replace v9.x and v12.x with your installed versions.)*

3. Copy the full path of this folder from the File Explorer address bar.

4. Add this path to your Windows system environment variables:
    - Search for `Edit the system environment variables` in the Start menu and open it.
    - Click on ```Environment Variables```
    - Under ```System Variables``` find and select ```Path```, then click ```Edit```
    - Click ```New``` and paste the copied `cuDNN` folder path.
    - Click OK to save and close all dialogs.
5. **Important:** Restart any open programs or reboot your computer to ensure Windows recognizes the 
new environment variable.

## TensorRT - Installation Steps
1. Download TensorRT from NVIDIA’s official site:\
    https://developer.nvidia.com/tensorrt

2. Click the `Download Now` button (you may need to log in with your NVIDIA account).

3. Select `TensorRT 10` and accept the terms and conditions.

4. Choose the latest release of `TensorRT 10`.

5. Download the Windows **ZIP package** for `CUDA 12.x`.

6. Once downloaded, unzip the TensorRT archive to a folder on your system.

7. Locate the `lib` folder inside the extracted TensorRT directory.

8. Copy the full path of this lib folder.

9. Add this folder path to your system’s `PATH environment variable`, following the same process as with CUDA (see CUDA installation step #4).

10. **Installation complete!** Restart any open programs or reboot your system to ensure the environment variables take effect.

# TensorRT Parameters in YoloDotNet
YoloDotNet provides a dedicated `TensorRtExecutionProvider` class with configurable parameters that allow you to tailor the TensorRT execution environment to your hardware and performance needs.
| Parameter                    | Description | Notes |
|------------------------------|-------------|-------|
| **GpuId**                    | The index of the NVIDIA GPU to use for inference. | `0` (default GPU) |
| **Precision**                | Numeric precision mode for inference:<br>`FP32` Full 32-bit float (highest accuracy)<BR>`FP16` Half precision for faster inference with minimal accuracy loss<br>`INT8` Integer precision offering fastest inference but requires calibration cache file | `FP32` is default
| **BuilderOptimizationLevel** | Controls the optimization effort when building the TensorRT engine cache. Higher values increase build time but may yield better runtime performance. Valid range: 0 to 5.| `3` is default
| **EngineCachePath**          | Directory path where TensorRT stores serialized engine cache files to speed up startup. | Must be set to a writable folder
| **EngineCachePrefix**        | Filename prefix for the engine cache files, helpful for distinguishing multiple model caches. | Optional. `YoloDotNet` is default
| **Int8CalibrationCacheFile** | Path to the calibration cache file required when using INT8 precision mode. This file enables TensorRT to correctly quantize the model. | Leave empty if not using `INT8`


### Tips for Usage
- Use `FP32` for maximum accuracy if speed is less critical.
- `FP16` provides a great balance of speed and accuracy on supported GPUs.
- `INT8` delivers the fastest inference but requires you to generate a calibration cache file beforehand. See the [INT8 Calibration Cache File](#int8-calibration-cache-file) section below.
- The engine cache avoids rebuilding the optimized engine on every run, dramatically improving startup times.
Always ensure your `EngineCachePath` is writable and managed (remove stale caches as needed).

### Engine Cache Files Management
TensorRT engine cache files stored in the `EngineCachePath` directory are not automatically deleted by **YoloDotNet** or **TensorRT**.
Whenever you update the model file or modify the TensorRT configuration (such as precision or optimization settings), a new engine cache file will be generated to reflect those changes.
The old cache files remain in the directory and will not be cleaned up automatically.
It is therefore the user’s responsibility to periodically review and remove outdated or unused cache files to avoid unnecessary disk usage and potential confusion.

# Initial Startup Delay (Engine Build Time)
When running YoloDotNet with TensorRT for the very first time, or after making changes to the inference configuration (e.g., precision, optimization level, model file, etc.), there is no existing engine cache that matches the new settings.

In this case, TensorRT will build a new optimized engine, which includes:

- Layer fusion
- Kernel auto-tuning
- Precision calibration
- Serialization of the engine to disk

⏳ This process can take anywhere from **several seconds to a few minutes**, depending on:

- Your GPU model and its compute capability
- The complexity and size of the ONNX model
- The selected precision mode (`FP32`, `FP16`, or `INT8`)
- The `BuilderOptimizationLevel` value

Once the engine cache file has been built and saved in the specified `EngineCachePath`, subsequent runs will load the prebuilt engine immediately, resulting in normal and fast startup times.

✅ **Tip:** This engine generation process only happens:

- If no matching cache exists.
- If the model or any relevant configuration changes.
- If the cache file was manually deleted.

📁 Always ensure that:
- `EngineCachePath` points to a writable and persistent folder.
- You preserve useful engine cache files for reuse in production environments.



## 🚫 Engine Cache Distribution Warning
The TensorRT engine cache file is hardware-specific. It is built and optimized based on:

- Your system architecture
- The installed CUDA and TensorRT version
- The exact GPU model and its compute capabilities
- Your `TensorRtExecutionProvider` configuration (e.g., precision, builder settings)

⚠️ Because of this, engine cache files are not portable and should not be distributed between machines or environments unless they are identical in hardware and software configuration.

Instead, it is recommended to allow TensorRT to build the engine cache at runtime on each target system. Once built, the engine will be reused on that specific machine for all future inferences.

#### ✅ Summary:
- Do not copy/share engine cache files across different machines.
- Let TensorRT build a cache specific to each target system.
- Use `EngineCachePrefix` if you want to support multiple engines side by side (e.g., for different models or configurations).

# INT8 Calibration Cache File
When using the INT8 precision mode in the `TensorRtExecutionProvider`, TensorRT requires a calibration cache file to accurately quantize your YOLO ONNX model. This cache contains important calibration data that enables TensorRT to perform `mixed precision` inference while maintaining model accuracy.

> ⚙️ Note:\
INT8 mode in TensorRT uses mixed precision execution. While it prioritizes INT8 for maximum performance, it will automatically fall back to FP16 or FP32 for layers that cannot be quantized due to unsupported operations, numerical stability concerns, or dynamic range limitations.

## Why is it needed?
- `INT8` inference offers the fastest performance but requires calibration because not all operations and layers can be safely quantized without losing accuracy.

- The calibration cache provides dynamic range information for tensors, allowing TensorRT to apply quantization selectively and reliably.

## How to generate the calibration cache file
You can create the calibration cache file using the `Ultralytics Python library` as follows:
1. Install the [Ultralytics library](https://docs.ultralytics.com/quickstart/#__tabbed_1_2)
2. Run the export command with INT8 calibration enabled:
    ```
    yolo export model=your_model.pt format=engine int8=true dynamic=true data=your_model_dataset.yaml opset=17
    ```
    Replace:
    - `your_model.pt` with the path to your PyTorch YOLO model.
    - `your_model_dataset.yaml` with your dataset YAML file.

3. Output:
    - `.engine` : The serialized TensorRT engine (can be ignored if not used).
    - `.cache` : The generated calibration cache file.

4. Use the cache in YoloDotNet
    ```csharp
    Int8CalibrationCacheFile = @"C:\path\to\your_model.cache";
    ```

## ⚠️ Important: Regenerating the Cache
TensorRT will not overwrite an existing calibration cache file.
To generate a new one, you must manually delete the old .cache file before running the export command again.

This ensures that calibration is performed from scratch using the new data or model configuration.

### ✅ Summary
- Required for INT8 precision inference in YoloDotNet.
- Created using the Ultralytics CLI and a calibration dataset.
- Must be manually removed if you want to regenerate it.
- Can be reused and shipped as part of your application.