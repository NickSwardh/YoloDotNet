# Benchmarks
This project includes a benchmarking suite to measure performance improvements and identify bottlenecks.

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

# YoloDotNet 3.0 Benchmark Results (Baseline)

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


# YoloDotNet 3.1.1 Benchmark Results (Current version)
```
BenchmarkDotNet v0.15.2, Windows 11 (10.0.26100.4652/24H2/2024Update/HudsonValley)
Intel Core i7-14700KF 3.40GHz, 1 CPU, 28 logical and 20 physical cores
.NET SDK 9.0.301
  [Host]     : .NET 8.0.17 (8.0.1725.26602), X64 RyuJIT AVX2
  DefaultJob : .NET 8.0.17 (8.0.1725.26602), X64 RyuJIT AVX2
```

##### CLASSIFICATION (Input image size: 1280x844)
| Method         | YoloParam     | Mean       | Error    | StdDev    | Gen0   | Allocated |
|--------------- |-------------- |-----------:|---------:|----------:|-------:|----------:|
| Classification | V8_Cls_CPU    | 2,662.5 us | 37.55 us |  35.12 us |      - |  32.32 KB |
| Classification | V8_Cls_GPU    |   831.4 us | 14.87 us |  17.13 us | 0.9766 |  32.32 KB |
| Classification | V8_Cls_TRT32  |   653.0 us | 10.85 us |  10.15 us | 0.9766 |  32.32 KB |
| Classification | V8_Cls_TRT16  |   459.6 us |  3.97 us |   3.31 us | 1.4648 |  32.32 KB |
| Classification | V8_Cls_TRT8   |   411.7 us |  3.54 us |   3.31 us | 1.4648 |  32.32 KB |
| Classification | V11_Cls_CPU   | 3,062.4 us | 56.81 us |  53.14 us |      - |  32.32 KB |
| Classification | V11_Cls_GPU   | 1,173.4 us | 41.59 us | 122.62 us |      - |  32.32 KB |
| Classification | V11_Cls_TRT32 |   765.1 us | 14.64 us |  16.86 us | 0.9766 |  32.32 KB |
| Classification | V11_Cls_TRT16 |   547.5 us |  4.97 us |   4.65 us | 0.9766 |  32.32 KB |
| Classification | V11_Cls_TRT8  |   485.3 us |  6.96 us |   6.17 us | 1.4648 |  32.32 KB |

##### OBJECT DETECTION (input image size: 1280x851)
| Method          | YoloParam     | Mean      | Error     | StdDev    | Median    | Allocated |
|---------------- |-------------- |----------:|----------:|----------:|----------:|----------:|
| ObjectDetection | V5u_Obj_CPU   | 29.441 ms | 0.5508 ms | 0.5152 ms | 29.324 ms |  31.85 KB |
| ObjectDetection | V5u_Obj_GPU   |  4.696 ms | 0.0214 ms | 0.0189 ms |  4.690 ms |  31.85 KB |
| ObjectDetection | V5u_Obj_TRT32 |  3.899 ms | 0.0762 ms | 0.0712 ms |  3.931 ms |  31.85 KB |
| ObjectDetection | V5u_Obj_TRT16 |  3.130 ms | 0.0126 ms | 0.0111 ms |  3.131 ms |  31.73 KB |
| ObjectDetection | V5u_Obj_TRT8  |  2.776 ms | 0.0104 ms | 0.0097 ms |  2.775 ms |  29.82 KB |
| ObjectDetection | V8_Obj_CPU    | 33.898 ms | 0.2453 ms | 0.2294 ms | 33.935 ms |  34.18 KB |
| ObjectDetection | V8_Obj_GPU    |  5.350 ms | 0.0133 ms | 0.0118 ms |  5.349 ms |  34.18 KB |
| ObjectDetection | V8_Obj_TRT32  |  4.061 ms | 0.0086 ms | 0.0072 ms |  4.061 ms |  34.08 KB |
| ObjectDetection | V8_Obj_TRT16  |  2.965 ms | 0.0138 ms | 0.0115 ms |  2.964 ms |  34.08 KB |
| ObjectDetection | V8_Obj_TRT8   |  2.739 ms | 0.0150 ms | 0.0141 ms |  2.742 ms |  35.28 KB |
| ObjectDetection | V9_Obj_CPU    | 38.662 ms | 0.5644 ms | 0.5003 ms | 38.689 ms |  25.66 KB |
| ObjectDetection | V9_Obj_GPU    |  8.705 ms | 0.1611 ms | 0.1507 ms |  8.765 ms |  25.44 KB |
| ObjectDetection | V9_Obj_TRT32  |  5.594 ms | 0.0333 ms | 0.0312 ms |  5.592 ms |  25.44 KB |
| ObjectDetection | V9_Obj_TRT16  |  4.195 ms | 0.0388 ms | 0.0344 ms |  4.198 ms |  25.89 KB |
| ObjectDetection | V9_Obj_TRT8   |  3.947 ms | 0.0390 ms | 0.0365 ms |  3.946 ms |  24.01 KB |
| ObjectDetection | V10_Obj_CPU   | 31.415 ms | 0.2898 ms | 0.2710 ms | 31.383 ms |  23.16 KB |
| ObjectDetection | V10_Obj_GPU   |  4.586 ms | 0.0534 ms | 0.0525 ms |  4.581 ms |  23.16 KB |
| ObjectDetection | V10_Obj_TRT32 |  3.233 ms | 0.0109 ms | 0.0102 ms |  3.232 ms |  23.16 KB |
| ObjectDetection | V10_Obj_TRT16 |  2.766 ms | 0.0414 ms | 0.0387 ms |  2.774 ms |  23.16 KB |
| ObjectDetection | V10_Obj_TRT8  |  2.551 ms | 0.0147 ms | 0.0130 ms |  2.547 ms |  23.51 KB |
| ObjectDetection | V11_Obj_CPU   | 31.552 ms | 0.6212 ms | 0.9106 ms | 31.419 ms |   30.7 KB |
| ObjectDetection | V11_Obj_GPU   |  5.827 ms | 0.0214 ms | 0.0179 ms |  5.831 ms |   30.7 KB |
| ObjectDetection | V11_Obj_TRT32 |  4.200 ms | 0.0710 ms | 0.0664 ms |  4.216 ms |   30.7 KB |
| ObjectDetection | V11_Obj_TRT16 |  3.264 ms | 0.0276 ms | 0.0258 ms |  3.272 ms |  30.57 KB |
| ObjectDetection | V11_Obj_TRT8  |  3.125 ms | 0.0623 ms | 0.1154 ms |  3.179 ms |  30.32 KB |
| ObjectDetection | V12_Obj_CPU   | 36.709 ms | 0.6856 ms | 0.8914 ms | 36.489 ms |  32.82 KB |
| ObjectDetection | V12_Obj_GPU   |  7.021 ms | 0.0500 ms | 0.0467 ms |  7.028 ms |  32.51 KB |
| ObjectDetection | V12_Obj_TRT32 |  4.864 ms | 0.0159 ms | 0.0149 ms |  4.862 ms |  32.51 KB |
| ObjectDetection | V12_Obj_TRT16 |  3.618 ms | 0.0454 ms | 0.0379 ms |  3.610 ms |  32.26 KB |
| ObjectDetection | V12_Obj_TRT8  |  3.644 ms | 0.0262 ms | 0.0219 ms |  3.639 ms |   31.1 KB |

##### ORIENTED OBJECT DETECTION (OBB) (input image size: 1280x720)
| Method       | YoloParam     | Mean      | Error     | StdDev    | Allocated |
|------------- |-------------- |----------:|----------:|----------:|----------:|
| ObbDetection | V8_Obb_CPU    | 92.457 ms | 1.7035 ms | 2.7019 ms |   8.15 KB |
| ObbDetection | V8_Obb_GPU    |  9.655 ms | 0.0450 ms | 0.0376 ms |   8.15 KB |
| ObbDetection | V8_Obb_TRT32  |  7.643 ms | 0.0151 ms | 0.0126 ms |   8.15 KB |
| ObbDetection | V8_Obb_TRT16  |  5.520 ms | 0.0370 ms | 0.0346 ms |   8.15 KB |
| ObbDetection | V8_Obb_TRT8   |  5.041 ms | 0.0191 ms | 0.0169 ms |   8.27 KB |
| ObbDetection | V11_Obb_CPU   | 86.191 ms | 1.7069 ms | 2.4479 ms |    7.9 KB |
| ObbDetection | V11_Obb_GPU   |  9.537 ms | 0.0300 ms | 0.0266 ms |    7.9 KB |
| ObbDetection | V11_Obb_TRT32 |  7.934 ms | 0.0429 ms | 0.0401 ms |    7.9 KB |
| ObbDetection | V11_Obb_TRT16 |  5.736 ms | 0.0166 ms | 0.0147 ms |    7.9 KB |
| ObbDetection | V11_Obb_TRT8  |  5.027 ms | 0.0161 ms | 0.0142 ms |   8.26 KB |

##### POSE ESTIMATION (input image size: 1280x720)
| Method         | YoloParam     | Mean      | Error     | StdDev    | Allocated |
|--------------- |-------------- |----------:|----------:|----------:|----------:|
| PoseEstimation | V8_Pos_CPU    | 35.362 ms | 0.7067 ms | 0.9434 ms |  23.81 KB |
| PoseEstimation | V8_Pos_GPU    |  5.246 ms | 0.0200 ms | 0.0167 ms |  23.81 KB |
| PoseEstimation | V8_Pos_TRT32  |  3.511 ms | 0.0535 ms | 0.0446 ms |  23.81 KB |
| PoseEstimation | V8_Pos_TRT16  |  2.424 ms | 0.0152 ms | 0.0135 ms |  23.81 KB |
| PoseEstimation | V8_Pos_TRT8   |  2.145 ms | 0.0215 ms | 0.0201 ms |  21.84 KB |
| PoseEstimation | V11_Pos_CPU   | 31.776 ms | 0.6348 ms | 0.7796 ms |  23.69 KB |
| PoseEstimation | V11_Pos_GPU   |  4.860 ms | 0.0147 ms | 0.0137 ms |  23.69 KB |
| PoseEstimation | V11_Pos_TRT32 |  3.540 ms | 0.0148 ms | 0.0138 ms |  23.69 KB |
| PoseEstimation | V11_Pos_TRT16 |  2.853 ms | 0.0260 ms | 0.0230 ms |  23.81 KB |
| PoseEstimation | V11_Pos_TRT8  |  2.259 ms | 0.0115 ms | 0.0204 ms |  24.44 KB |

##### SEGMENTATION (input image size: 1280x853)
| Method       | YoloParam     | Mean      | Error     | StdDev    | Allocated |
|------------- |-------------- |----------:|----------:|----------:|----------:|
| Segmentation | V8_Seg_CPU    | 47.793 ms | 0.7903 ms | 0.7392 ms |  84.92 KB |
| Segmentation | V8_Seg_GPU    |  7.373 ms | 0.1047 ms | 0.0979 ms |  84.92 KB |
| Segmentation | V8_Seg_TRT32  |  6.259 ms | 0.0937 ms | 0.0876 ms |  84.92 KB |
| Segmentation | V8_Seg_TRT16  |  4.886 ms | 0.0505 ms | 0.0519 ms |  84.89 KB |
| Segmentation | V11_Seg_CPU   | 44.502 ms | 0.8490 ms | 0.7941 ms |  79.53 KB |
| Segmentation | V11_Seg_GPU   |  7.805 ms | 0.1083 ms | 0.0904 ms |  79.53 KB |
| Segmentation | V11_Seg_TRT32 |  6.363 ms | 0.0287 ms | 0.0255 ms |  79.53 KB |
| Segmentation | V11_Seg_TRT16 |  5.315 ms | 0.1049 ms | 0.1571 ms |  79.58 KB |
>Note: Segmentation running on TensorRT with INT8 precision is not supported.