# YoloDotNet.ExecutionProvider.MIGraphX

MIGraphX execution provider for YoloDotNet, enabling GPU-accelerated inference on AMD ROCm hardware.

## Overview

This execution provider enables high-performance YOLO inference on AMD GPUs using the MIGraphX execution provider from ONNX Runtime. It supports AMD Radeon and Instinct GPUs with ROCm on Linux.

## Requirements

- **Operating System**: Linux (ROCm support)
- **Hardware**: AMD Radeon or Instinct GPU with ROCm support
- **ROCm**: Version 5.0 or higher
- **.NET**: .NET 8.0 or higher
- **YoloDotNet**: Version 4.1 or higher
- **ONNX Runtime**: MIGraphX execution provider is included in the standard `Microsoft.ML.OnnxRuntime` package

## Installation

```bash
dotnet add package YoloDotNet.ExecutionProvider.MIGraphX
```

**Note**: This package uses `Microsoft.ML.OnnxRuntime` which includes MIGraphX execution provider support. You need to have ROCm installed on your Linux system to use AMD GPU acceleration.

## Usage

```csharp
using YoloDotNet;
using YoloDotNet.ExecutionProvider.MIGraphX;

// Create MIGraphX execution provider
var provider = new MIGraphXExecutionProvider("path/to/model.onnx", deviceId: 0);

// Use with YoloDotNet
var yolo = new Yolo(provider);

// Run inference
var results = yolo.RunObjectDetection("image.jpg", confidence: 0.25, iou: 0.7);
```

## Configuration Options

### Device Selection

```csharp
// Use first GPU (default)
var provider = new MIGraphXExecutionProvider("model.onnx", deviceId: 0);

// Use second GPU
var provider = new MIGraphXExecutionProvider("model.onnx", deviceId: 1);

// Use CPU fallback
var provider = new MIGraphXExecutionProvider("model.onnx", deviceId: -1);
```

## Performance Tips

1. **FP16 Precision**: MIGraphX automatically enables FP16 for better performance on supported hardware
2. **Model Optimization**: Use ONNX models optimized for your specific AMD GPU architecture
3. **Batch Processing**: Process multiple images in batches for better GPU utilization
4. **ROCm Version**: Use the latest ROCm version for best performance and compatibility

## Supported YOLO Models

- YOLOv5u
- YOLOv8
- YOLOv9
- YOLOv10
- YOLOv11
- YOLOv12
- YOLOv26
- YOLO-World
- YOLO-E
- RT-DETR

## Supported Tasks

- Object Detection
- Instance Segmentation
- Image Classification
- Pose Estimation
- Oriented Bounding Box (OBB) Detection

## Troubleshooting

### ROCm Not Found

Ensure ROCm is properly installed and configured:

```bash
rocm-smi  # Check ROCm installation
```

### GPU Not Detected

Verify your GPU is supported by ROCm:

```bash
rocminfo  # List available ROCm devices
```

### Performance Issues

- Update to the latest ROCm drivers
- Ensure your model is in ONNX format (not PyTorch .pt)
- Use FP16 models for better performance
- Check GPU memory usage with `rocm-smi`

## License

MIT License - Copyright © Niklas Swärd

## Links

- [YoloDotNet GitHub](https://github.com/NickSwardh/YoloDotNet)
- [AMD ROCm Documentation](https://rocm.docs.amd.com/)
- [ONNX Runtime MIGraphX EP](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html)
