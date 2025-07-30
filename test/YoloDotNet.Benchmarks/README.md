# Benchmarks
This project includes a benchmarking suite to measure performance improvements and identify bottlenecks.

### Summary of Improvements

> **Note:** This summary highlights the *maximum* observed speed and memory improvements between versions.
> For full benchmark results, hardware used, detailed per-task analysis, and raw data, please see the complete benchmarks at the end of this page.

| YoloDotNet v3.0     | Device | Speed Gain         | Memory Reduction        |
| ------------------- | ------ | ------------------ | ----------------------- |
| **Segmentation**    | GPU    | up to 70.8% faster | up to 92.7% less memory |
|                     | CPU    | up to 12.6% faster | up to 92.7% less memory |
| **Classification**  | GPU    | up to 28.5% faster | up to 46.1% less memory |
|                     | CPU    | up to 9.6% faster  | up to 46.1% less memory |
| **OBB Detection**   | GPU    | up to 28.7% faster | up to 2.2% less memory  |
|                     | CPU    | up to 4.7% faster  | up to 2.9% less memory  |
| **Pose Estimation** | GPU    | up to 27.6% faster | up to 0.7% less memory  |
|                     | CPU    | up to 4.6% faster  | up to 0.7% less memory  |
| **Detection**       | GPU    | up to 25.0% faster | up to 0.8% less memory  |
|                     | CPU    | up to 5.3% faster  | up to 1.0% less memory  |

*Benchmarks performed using NVIDIA GeForce RTX 4070 Ti GPU and Intel Core i7-14700KF*

## Benchmark Models and Images
The benchmarks performed in this project use the same [models](../assets/Models/) and [images](../assets/Media/) as those included in the YoloDotNet demo projects. This ensures consistency and transparency when evaluating performance metrics across different execution providers.

### Running Benchmarks
- Build the solution in `Release` mode.
- Run the `YoloDotNet.Benchmarks` project.

> **Note**: While it is possible to run benchmarks in Debug mode by uncommenting specific sections, this approach is not recommended for obtaining accurate performance data. Debug mode should be reserved
primarily for stepping through and diagnosing the benchmark code.

### Benchmark Results
- Benchmark results are presented as **baseline (previous version)** and **current version** to facilitate easy comparison of performance improvements.

- Please note that benchmark outcomes depend heavily on hardware and environmental factors. To obtain meaningful and consistent results, it is recommended to run benchmarks on the same hardware configuration whenever possible.

Keep in mind that differences in hardware, system load, driver versions, and other factors can influence benchmark performance. Use these results as a general guide rather than absolute metrics.

### Hardware Used for Benchmarks
`NVIDIA GeForce RTX 4070 Ti GPU`

# YoloDotNet 2.3 Benchmark Results (Baseline)

```
BenchmarkDotNet v0.14.0, Windows 11 (10.0.26100.3476)
Intel Core i7-14700KF, 1 CPU, 28 logical and 20 physical cores
.NET SDK 9.0.103
  [Host]     : .NET 8.0.13 (8.0.1325.6609), X64 RyuJIT AVX2
  DefaultJob : .NET 8.0.13 (8.0.1325.6609), X64 RyuJIT AVX2
```

##### CLASSIFICATION (Input image size: 1280x844)
| Method                    | Mean     | Error     | StdDev    | Median   | Gen0   | Allocated | Model Used
|-------------------------- |---------:|----------:|----------:|---------:|-------:|----------:|--------------:
| ClassificationYolov8Cpu   | 2.990 ms | 0.0475 ms | 0.0444 ms | 2.989 ms |      - |  59.92 KB | yolov8s-cls  |
| ClassificationYolov8Gpu   | 1.157 ms | 0.0392 ms | 0.1156 ms | 1.091 ms | 1.9531 |  59.92 KB | yolov8s-cls  |
| ClassificationYolov11Cpu  | 3.313 ms | 0.0637 ms | 0.0914 ms | 3.321 ms |      - |  59.92 KB | yolov11s-cls |
| ClassificationYolov11Gpu  | 1.252 ms | 0.0038 ms | 0.0035 ms | 1.253 ms | 1.9531 |  59.92 KB | yolov11s-cls |

##### OBJECT DETECTION (input image size: 1280x851)
| Method                    | Mean      | Error     | StdDev    | Allocated | Model Used
|-------------------------- |----------:|----------:|----------:|----------:|----------:
| ObjectDetectionYolov8Cpu  | 34.893 ms | 0.4399 ms | 0.4115 ms |  34.52 KB | yolov8s  |
| ObjectDetectionYolov8Gpu  |  6.670 ms | 0.0152 ms | 0.0142 ms |  34.47 KB | yolov8s  |
| ObjectDetectionYolov9Cpu  | 39.623 ms | 0.7737 ms | 1.0590 ms |  29.34 KB | yolov9s  |
| ObjectDetectionYolov9Gpu  | 10.037 ms | 0.1195 ms | 0.1117 ms |  29.32 KB | yolov9s  |
| ObjectDetectionYolov10Cpu | 32.120 ms | 0.6222 ms | 0.7406 ms |   24.4 KB | yolov10s |
| ObjectDetectionYolov10Gpu |  6.571 ms | 0.0377 ms | 0.0334 ms |  24.35 KB | yolov10s |
| ObjectDetectionYolov11Cpu | 32.133 ms | 0.6097 ms | 0.6524 ms |  32.62 KB | yolov11s |
| ObjectDetectionYolov11Gpu |  6.736 ms | 0.0241 ms | 0.0213 ms |  32.57 KB | yolov11s |
| ObjectDetectionYolov12Cpu | 39.184 ms | 0.7626 ms | 0.9644 ms |     31 KB | yolov12s |
| ObjectDetectionYolov12Gpu |  9.046 ms | 0.1107 ms | 0.1035 ms |  30.95 KB | yolov12s |

##### ORIENTED OBJECT DETECTION (OBB) (input image size: 1280x720)
| Method                 | Mean     | Error    | StdDev   | Allocated | Model Used
|----------------------- |---------:|---------:|---------:|----------:|--------------:
| ObbDetectionYolov8Cpu  | 93.61 ms | 1.086 ms | 0.963 ms |   8.39 KB | yolov8s-obb  |
| ObbDetectionYolov8Gpu  | 13.31 ms | 0.052 ms | 0.049 ms |   8.33 KB | yolov8s-obb  |
| ObbDetectionYolov11Cpu | 85.04 ms | 1.683 ms | 1.653 ms |   8.39 KB | yolov11s-obb |
| ObbDetectionYolov11Gpu | 13.27 ms | 0.060 ms | 0.056 ms |   8.33 KB | yolov11s-obb |

##### POSE ESTIMATION (input image size: 1280x720)
| Method                   | Mean      | Error     | StdDev    | Allocated | Model Used
|------------------------- |----------:|----------:|----------:|----------:|---------------:
| PoseEstimationYolov8Cpu  | 36.508 ms | 0.3856 ms | 0.3418 ms |  23.97 KB | yolov8s-pose  |
| PoseEstimationYolov8Gpu  |  6.617 ms | 0.0271 ms | 0.0254 ms |  23.97 KB | yolov8s-pose  |
| PoseEstimationYolov11Cpu | 33.325 ms | 0.5708 ms | 0.5339 ms |  21.98 KB | yolov11s-pose |
| PoseEstimationYolov11Gpu |  6.458 ms | 0.0307 ms | 0.0272 ms |  21.98 KB | yolov11s-pose |

##### SEGMENTATION (input image size: 1280x853)
| Method                 | Mean     | Error    | StdDev   | Gen0     | Gen1     | Gen2     | Allocated | Model Used
|----------------------- |---------:|---------:|---------:|---------:|---------:|---------:|----------:|--------------:
| SegmentationYolov8Cpu  | 60.07 ms | 1.050 ms | 0.983 ms | 444.4444 | 333.3333 | 111.1111 |   7.52 MB | yolov8s-seg  |
| SegmentationYolov8Gpu  | 34.93 ms | 0.692 ms | 1.963 ms | 468.7500 | 437.5000 | 156.2500 |   7.51 MB | yolov8s-seg  |
| SegmentationYolov11Cpu | 55.90 ms | 1.101 ms | 1.353 ms | 444.4444 | 333.3333 | 111.1111 |   7.01 MB | yolov11s-seg |
| SegmentationYolov11Gpu | 24.23 ms | 0.306 ms | 0.286 ms | 468.7500 | 437.5000 | 156.2500 |   7.01 MB | yolov11s-seg |


# YoloDotNet 3.0 Benchmark Results (Current version)
```
BenchmarkDotNet v0.15.2, Windows 11 (10.0.26100.4652/24H2/2024Update/HudsonValley)
Intel Core i7-14700KF 3.40GHz, 1 CPU, 28 logical and 20 physical cores
.NET SDK 9.0.301
  [Host]     : .NET 8.0.17 (8.0.1725.26602), X64 RyuJIT AVX2
  DefaultJob : .NET 8.0.17 (8.0.1725.26602), X64 RyuJIT AVX2
```

##### CLASSIFICATION (Input image size: 1280x844)
| Method         | YoloParam   | Mean       | Error    | StdDev   | Gen0   | Allocated |
|--------------- |------------ |-----------:|---------:|---------:|-------:|----------:|
| Classification | V8_Cls_CPU  | 2,703.9 us | 51.32 us | 50.41 us |      - |  32.32 KB |
| Classification | V8_Cls_GPU  |   826.8 us | 15.40 us | 14.40 us | 0.9766 |  32.32 KB |
| Classification | V11_Cls_CPU | 3,075.7 us | 61.02 us | 65.29 us |      - |  32.32 KB |
| Classification | V11_Cls_GPU | 1,007.8 us |  4.72 us |  4.41 us |      - |  32.32 KB |

##### OBJECT DETECTION (input image size: 1280x851)
| Method          | YoloParam   | Mean      | Error     | StdDev    | Allocated |
|---------------- |------------ |----------:|----------:|----------:|----------:|
| ObjectDetection | V5u_Obj_CPU | 29.169 ms | 0.5444 ms | 0.8791 ms |  31.85 KB |
| ObjectDetection | V5u_Obj_GPU |  4.659 ms | 0.0313 ms | 0.0293 ms |  31.85 KB |
| ObjectDetection | V8_Obj_CPU  | 33.040 ms | 0.5668 ms | 0.4733 ms |  34.18 KB |
| ObjectDetection | V8_Obj_GPU  |  5.001 ms | 0.0321 ms | 0.0285 ms |  34.18 KB |
| ObjectDetection | V9_Obj_CPU  | 38.676 ms | 0.7520 ms | 1.2769 ms |  25.66 KB |
| ObjectDetection | V9_Obj_GPU  |  6.711 ms | 0.0330 ms | 0.0309 ms |  25.44 KB |
| ObjectDetection | V10_Obj_CPU | 30.650 ms | 0.2735 ms | 0.2558 ms |  19.33 KB |
| ObjectDetection | V10_Obj_GPU |  5.206 ms | 0.0458 ms | 0.0428 ms |  19.19 KB |
| ObjectDetection | V11_Obj_CPU | 30.662 ms | 0.3778 ms | 0.3534 ms |  30.84 KB |
| ObjectDetection | V11_Obj_GPU |  4.948 ms | 0.0214 ms | 0.0190 ms |   30.7 KB |
| ObjectDetection | V12_Obj_CPU | 30.601 ms | 0.3743 ms | 0.3501 ms |  30.84 KB |
| ObjectDetection | V12_Obj_GPU |  5.050 ms | 0.0319 ms | 0.0266 ms |   30.7 KB |

##### ORIENTED OBJECT DETECTION (OBB) (input image size: 1280x720)
| Method       | YoloParam   | Mean      | Error     | StdDev    | Allocated |
|------------- |------------ |----------:|----------:|----------:|----------:|
| ObbDetection | V8_Obb_CPU  | 89.218 ms | 0.9932 ms | 0.9291 ms |   8.15 KB |
| ObbDetection | V8_Obb_GPU  |  9.492 ms | 0.0575 ms | 0.0538 ms |   8.15 KB |
| ObbDetection | V11_Obb_CPU | 81.068 ms | 0.5582 ms | 0.4948 ms |    7.9 KB |
| ObbDetection | V11_Obb_GPU |  9.622 ms | 0.0508 ms | 0.0475 ms |    7.9 KB |

##### POSE ESTIMATION (input image size: 1280x720)
| Method         | YoloParam   | Mean      | Error     | StdDev    | Allocated |
|--------------- |------------ |----------:|----------:|----------:|----------:|
| PoseEstimation | V8_Pos_CPU  | 34.846 ms | 0.2279 ms | 0.2132 ms |  23.81 KB |
| PoseEstimation | V8_Pos_GPU  |  4.793 ms | 0.0258 ms | 0.0215 ms |  23.81 KB |
| PoseEstimation | V11_Pos_CPU | 31.511 ms | 0.3839 ms | 0.3591 ms |  23.69 KB |
| PoseEstimation | V11_Pos_GPU |  5.662 ms | 0.1122 ms | 0.1049 ms |  23.69 KB |

##### SEGMENTATION (input image size: 1280x853)
| Method       | YoloParam   | Mean     | Error    | StdDev   | Median   | Gen0    | Allocated |
|------------- |------------ |---------:|---------:|---------:|---------:|--------:|----------:|
| Segmentation | V8_Seg_CPU  | 52.52 ms | 1.015 ms | 0.949 ms | 52.43 ms |       - | 547.73 KB |
| Segmentation | V8_Seg_GPU  | 10.21 ms | 0.020 ms | 0.017 ms | 10.21 ms | 31.2500 | 547.73 KB |
| Segmentation | V11_Seg_CPU | 50.20 ms | 1.088 ms | 3.207 ms | 49.11 ms |       - | 533.65 KB |
| Segmentation | V11_Seg_GPU | 10.53 ms | 0.213 ms | 0.306 ms | 10.69 ms | 31.2500 | 533.65 KB |