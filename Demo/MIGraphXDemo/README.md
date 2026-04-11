# MIGraphX Demo

This demo showcases YoloDotNet running with AMD MIGraphX execution provider for GPU-accelerated inference on AMD hardware with ROCm.

## Requirements

- **Operating System**: Linux with ROCm support
- **Hardware**: AMD Radeon or Instinct GPU
- **ROCm**: Version 5.0 or higher
- **.NET**: .NET 8.0 SDK

## Setup

### 1. Install ROCm

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install rocm-dev

# Verify installation
rocm-smi
rocminfo
```

### 2. Build the demo

```bash
cd Demo/MIGraphXDemo
dotnet build
```

## Usage

```bash
dotnet run --project MIGraphXDemo.csproj <model.onnx> <image.jpg> [deviceId]
```

### Parameters

- `model.onnx` - Path to your ONNX YOLO model
- `image.jpg` - Path to input image
- `deviceId` - (Optional) GPU device ID (default: 0, use -1 for CPU fallback)

### Examples

```bash
# Use first AMD GPU (default)
dotnet run yolov8n.onnx image.jpg

# Use second AMD GPU
dotnet run yolov8n.onnx image.jpg 1

# CPU fallback (no GPU)
dotnet run yolov8n.onnx image.jpg -1
```

## What This Demo Does

1. ✅ Loads ONNX model with MIGraphX execution provider
2. ✅ Displays model information (type, input shape, data type)
3. ✅ Runs object detection inference
4. ✅ Shows detected objects with confidence scores
5. ✅ Draws bounding boxes on the image
6. ✅ Saves result image
7. ✅ Runs benchmark (10 iterations) and shows performance metrics

## Expected Output

```
YoloDotNet - MIGraphX (AMD ROCm) Demo
=====================================

Model: yolov8n.onnx
Image: image.jpg
Device ID: 0 (AMD GPU)

Initializing MIGraphX execution provider...
Model loaded: ObjectDetection
Input shape: 1x3x640x640
Data type: Float

Loading image...
Image size: 1920x1080

Running inference...
Inference completed in 45ms
Detections: 12

Detected objects:
  - person: 95.23% at (120, 340)
  - car: 89.45% at (450, 200)
  - dog: 87.12% at (800, 500)
  ...

Drawing results...
Result saved to: image_migraphx_result.jpg

Running benchmark (10 iterations)...
Average: 42.30ms
Min: 40ms
Max: 45ms

✓ Demo completed successfully!
```

## Troubleshooting

### ROCm Not Found

```bash
# Check ROCm installation
rocm-smi

# If not installed, install ROCm
sudo apt install rocm-dev
```

### GPU Not Detected

```bash
# List available ROCm devices
rocminfo

# Check GPU status
rocm-smi
```

### Performance Issues

- Update ROCm to the latest version
- Use FP16 models for better performance
- Ensure no other processes are using the GPU
- Check GPU memory usage: `rocm-smi`

### Model Loading Errors

- Verify ONNX model format
- Check opset version (17 for YOLOv8-v12, 18 for YOLOv26)
- Try CPU fallback: `deviceId=-1`

## Performance Tips

1. **Use FP16 models** - MIGraphX automatically enables FP16 optimization
2. **Batch processing** - Process multiple images for better GPU utilization
3. **Model optimization** - Use ONNX models optimized for your GPU architecture
4. **Latest ROCm** - Keep ROCm updated for best performance

## Supported Models

All YoloDotNet-compatible models work with MIGraphX:

- YOLOv5u, YOLOv8, YOLOv9, YOLOv10, YOLOv11, YOLOv12, YOLOv26
- YOLO-World, YOLO-E, RT-DETR
- All tasks: Detection, Segmentation, Classification, Pose, OBB

## Links

- [YoloDotNet GitHub](https://github.com/NickSwardh/YoloDotNet)
- [AMD ROCm Documentation](https://rocm.docs.amd.com/)
- [ONNX Runtime MIGraphX EP](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html)
