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
- Benchmark outcomes depend heavily on hardware and environmental factors. To obtain meaningful and consistent results, it is recommended to run benchmarks on the same hardware configuration whenever possible.

- The results below were produced on a **different computer** than the results published for earlier versions (YoloDotNet 3.0 and 3.1.1). Numbers from earlier versions are therefore **not directly comparable** and have been removed.

- These results include **CUDA Graph** benchmarks in addition to the regular CPU, GPU (CUDA) and TensorRT (FP32/FP16) runs.

Keep in mind that differences in hardware, system load, driver versions, and other factors can influence benchmark performance. Use these results as a general guide rather than absolute metrics.

### Model Column Legend
| Label                   | Meaning                                              |
|-------------------------|------------------------------------------------------|
| `CPU`                   | Ran on the CPU                                       |
| `GPU`                   | Ran on the GPU using the CUDA execution provider     |
| `GPU TensorRT FP32`     | Ran on the GPU using TensorRT, FP32 precision        |
| `GPU TensorRT FP16`     | Ran on the GPU using TensorRT, FP16 precision        |
| `+ CUDA Graph`          | Same as above, with CUDA Graph enabled               |

# YoloDotNet 4.3 Benchmark Results

```
BenchmarkDotNet v0.15.8, Windows 11 (10.0.26300.9711)
AMD Ryzen 7 9800X3D 4.70GHz, 1 CPU, 16 logical and 8 physical cores
.NET SDK 10.0.403
  [Host]     : .NET 8.0.33 (8.0.33, 8.0.3326.52219), X64 RyuJIT x86-64-v4
  DefaultJob : .NET 8.0.33 (8.0.33, 8.0.3326.52219), X64 RyuJIT x86-64-v4
```

## Results Overview
Select a task below to jump directly to its benchmark results.

| Task                                                                       | Models                                                              | Test image                                           |
|----------------------------------------------------------------------------|---------------------------------------------------------------------|------------------------------------------------------|
| [Classification](#classification-input-image-size-1280x844)                | YOLOv8, YOLO11, YOLO26                                              | [hummingbird.jpg](../assets/Media/hummingbird.jpg)   |
| [Object Detection](#object-detection-input-image-size-1280x851)            | YOLOv5u, YOLOv8, YOLOv9, YOLOv10, YOLO11, YOLO12, YOLO26            | [street.jpg](../assets/Media/street.jpg)             |
| [Oriented Object Detection (OBB)](#oriented-object-detection-obb-input-image-size-1280x720) | YOLOv8, YOLO11, YOLO26                             | [island.jpg](../assets/Media/island.jpg)             |
| [Pose Estimation](#pose-estimation-input-image-size-1280x720)              | YOLOv8, YOLO11, YOLO26                                              | [crosswalk.jpg](../assets/Media/crosswalk.jpg)       |
| [Segmentation](#segmentation-input-image-size-1280x853)                    | YOLOv8, YOLO11, YOLO26                                              | [people.jpg](../assets/Media/people.jpg)             |

##### CLASSIFICATION (Input image size: 1280x844)
Image: [hummingbird.jpg](../assets/Media/hummingbird.jpg)

| Model                                       | Mean       | Error    | StdDev   | Allocated |
|-------------------------------------------- |-----------:|---------:|---------:|----------:|
| YOLOv8 - CPU                                | 1,256.4 us | 25.06 us | 65.58 us |    1120 B |
| YOLOv8 - GPU                                | 1,010.1 us |  3.50 us |  2.92 us |    1120 B |
| YOLOv8 - GPU TensorRT FP32                  |   639.5 us |  2.34 us |  2.19 us |    1120 B |
| YOLOv8 - GPU TensorRT FP16                  |   474.0 us |  0.99 us |  0.93 us |    1120 B |
| YOLO11 - CPU                                | 1,205.4 us | 24.08 us | 32.96 us |    1120 B |
| YOLO11 - GPU                                | 1,282.3 us |  3.47 us |  3.08 us |    1120 B |
| YOLO11 - GPU TensorRT FP32                  |   779.8 us |  4.16 us |  3.89 us |    1120 B |
| YOLO11 - GPU TensorRT FP16                  |   598.1 us |  7.00 us |  6.55 us |    1120 B |
| YOLO26 - CPU                                | 1,176.8 us | 16.97 us | 15.04 us |    1120 B |
| YOLO26 - GPU                                | 1,296.8 us | 10.15 us |  9.49 us |    1120 B |
| YOLO26 - GPU TensorRT FP32                  |   777.1 us |  3.50 us |  3.27 us |    1120 B |
| YOLO26 - GPU TensorRT FP16                  |   593.3 us |  4.67 us |  4.37 us |    1120 B |
| YOLOv8 - GPU + CUDA Graph                   |   690.5 us |  1.78 us |  1.67 us |     528 B |
| YOLO11 - GPU + CUDA Graph                   |   844.6 us |  2.37 us |  2.22 us |     528 B |
| YOLO26 - GPU + CUDA Graph                   |   832.0 us |  1.60 us |  1.50 us |     528 B |
| YOLOv8 - GPU TensorRT FP32 + CUDA Graph     |   475.8 us |  5.04 us |  4.72 us |     528 B |
| YOLOv8 - GPU TensorRT FP16 + CUDA Graph     |   301.0 us |  1.47 us |  1.31 us |     528 B |
| YOLO11 - GPU TensorRT FP32 + CUDA Graph     |   513.3 us |  2.35 us |  2.20 us |     528 B |
| YOLO11 - GPU TensorRT FP16 + CUDA Graph     |   342.3 us |  1.38 us |  1.29 us |     528 B |
| YOLO26 - GPU TensorRT FP32 + CUDA Graph     |   514.7 us |  2.98 us |  2.78 us |     528 B |
| YOLO26 - GPU TensorRT FP16 + CUDA Graph     |   347.1 us |  0.73 us |  0.61 us |     528 B |

##### OBJECT DETECTION (input image size: 1280x851)
Image: [street.jpg](../assets/Media/street.jpg)

| Model                                       | Mean      | Error     | StdDev    | Allocated |
|-------------------------------------------- |----------:|----------:|----------:|----------:|
| YOLOv5u - CPU                               | 15.226 ms | 0.2911 ms | 0.6011 ms |  29.07 KB |
| YOLOv5u - GPU                               |  4.002 ms | 0.0085 ms | 0.0079 ms |  29.09 KB |
| YOLOv5u - GPU TensorRT FP32                 |  2.784 ms | 0.0040 ms | 0.0037 ms |  29.07 KB |
| YOLOv5u - GPU TensorRT FP16                 |  1.870 ms | 0.0088 ms | 0.0082 ms |  28.93 KB |
| YOLOv8 - CPU                                | 17.294 ms | 0.3395 ms | 0.8264 ms |  30.62 KB |
| YOLOv8 - GPU                                |  3.993 ms | 0.0059 ms | 0.0053 ms |   30.7 KB |
| YOLOv8 - GPU TensorRT FP32                  |  2.833 ms | 0.0096 ms | 0.0090 ms |  30.59 KB |
| YOLOv8 - GPU TensorRT FP16                  |  1.916 ms | 0.0072 ms | 0.0067 ms |  30.56 KB |
| YOLOv9 - CPU                                | 17.970 ms | 0.3290 ms | 0.6339 ms |  26.83 KB |
| YOLOv9 - GPU                                |  7.247 ms | 0.0209 ms | 0.0195 ms |  26.86 KB |
| YOLOv9 - GPU TensorRT FP32                  |  4.113 ms | 0.0220 ms | 0.0206 ms |   26.9 KB |
| YOLOv9 - GPU TensorRT FP16                  |  2.881 ms | 0.0149 ms | 0.0139 ms |  26.95 KB |
| YOLOv10 - CPU                               | 14.741 ms | 0.2929 ms | 0.4894 ms |  10.39 KB |
| YOLOv10 - GPU                               |  4.002 ms | 0.0054 ms | 0.0051 ms |  10.41 KB |
| YOLOv10 - GPU TensorRT FP32                 |  2.639 ms | 0.0051 ms | 0.0048 ms |  10.38 KB |
| YOLOv10 - GPU TensorRT FP16                 |  1.814 ms | 0.0089 ms | 0.0083 ms |   10.4 KB |
| YOLO11 - CPU                                | 14.576 ms | 0.2816 ms | 0.3759 ms |  28.76 KB |
| YOLO11 - GPU                                |  4.384 ms | 0.0087 ms | 0.0077 ms |  28.62 KB |
| YOLO11 - GPU TensorRT FP32                  |  3.051 ms | 0.0104 ms | 0.0093 ms |  28.62 KB |
| YOLO11 - GPU TensorRT FP16                  |  2.109 ms | 0.0092 ms | 0.0081 ms |  28.61 KB |
| YOLO12 - CPU                                | 17.132 ms | 0.3406 ms | 0.6312 ms |  28.25 KB |
| YOLO12 - GPU                                |  5.350 ms | 0.0103 ms | 0.0096 ms |  28.23 KB |
| YOLO12 - GPU TensorRT FP32                  |  3.732 ms | 0.0196 ms | 0.0164 ms |  28.06 KB |
| YOLO12 - GPU TensorRT FP16                  |  2.764 ms | 0.0096 ms | 0.0085 ms |  28.12 KB |
| YOLO26 - CPU                                | 13.555 ms | 0.2701 ms | 0.5268 ms |   7.15 KB |
| YOLO26 - GPU                                |  4.551 ms | 0.0123 ms | 0.0115 ms |   7.16 KB |
| YOLO26 - GPU TensorRT FP32                  |  3.004 ms | 0.0044 ms | 0.0039 ms |    7.2 KB |
| YOLO26 - GPU TensorRT FP16                  |  1.986 ms | 0.0138 ms | 0.0130 ms |   7.15 KB |
| YOLOv5u - GPU + CUDA Graph                  |  3.104 ms | 0.0028 ms | 0.0027 ms |  28.49 KB |
| YOLOv8 - GPU + CUDA Graph                   |  3.134 ms | 0.0175 ms | 0.0164 ms |  30.14 KB |
| YOLOv9 - GPU + CUDA Graph                   |  4.689 ms | 0.0102 ms | 0.0096 ms |   26.3 KB |
| YOLOv10 - GPU + CUDA Graph                  |  3.004 ms | 0.0141 ms | 0.0132 ms |    9.9 KB |
| YOLO11 - GPU + CUDA Graph                   |  3.321 ms | 0.0129 ms | 0.0120 ms |  28.07 KB |
| YOLO12 - GPU + CUDA Graph                   |  4.012 ms | 0.0087 ms | 0.0081 ms |  27.68 KB |
| YOLO26 - GPU + CUDA Graph                   |  3.351 ms | 0.0105 ms | 0.0098 ms |   6.62 KB |
| YOLOv5u - GPU TensorRT FP32 + CUDA Graph    |  2.239 ms | 0.0432 ms | 0.0404 ms |  28.47 KB |
| YOLOv5u - GPU TensorRT FP16 + CUDA Graph    |  1.375 ms | 0.0206 ms | 0.0193 ms |  28.34 KB |
| YOLOv8 - GPU TensorRT FP32 + CUDA Graph     |  2.464 ms | 0.0060 ms | 0.0057 ms |     30 KB |
| YOLOv8 - GPU TensorRT FP16 + CUDA Graph     |  1.489 ms | 0.0092 ms | 0.0086 ms |  29.98 KB |
| YOLOv9 - GPU TensorRT FP32 + CUDA Graph     |  3.231 ms | 0.0099 ms | 0.0092 ms |  26.36 KB |
| YOLOv9 - GPU TensorRT FP16 + CUDA Graph     |  1.916 ms | 0.0048 ms | 0.0045 ms |  26.32 KB |
| YOLOv10 - GPU TensorRT FP32 + CUDA Graph    |  2.041 ms | 0.0046 ms | 0.0043 ms |   9.84 KB |
| YOLOv10 - GPU TensorRT FP16 + CUDA Graph    |  1.204 ms | 0.0223 ms | 0.0209 ms |   9.74 KB |
| YOLO11 - GPU TensorRT FP32 + CUDA Graph     |  2.443 ms | 0.0074 ms | 0.0069 ms |  28.05 KB |
| YOLO11 - GPU TensorRT FP16 + CUDA Graph     |  1.519 ms | 0.0110 ms | 0.0103 ms |  28.01 KB |
| YOLO12 - GPU TensorRT FP32 + CUDA Graph     |  2.850 ms | 0.0088 ms | 0.0082 ms |  27.49 KB |
| YOLO12 - GPU TensorRT FP16 + CUDA Graph     |  1.909 ms | 0.0043 ms | 0.0040 ms |   27.6 KB |
| YOLO26 - GPU TensorRT FP32 + CUDA Graph     |  2.323 ms | 0.0100 ms | 0.0093 ms |   6.59 KB |
| YOLO26 - GPU TensorRT FP16 + CUDA Graph     |  1.309 ms | 0.0159 ms | 0.0149 ms |   6.56 KB |

##### ORIENTED OBJECT DETECTION (OBB) (input image size: 1280x720)
Image: [island.jpg](../assets/Media/island.jpg)

| Model                                       | Mean      | Error     | StdDev    | Allocated |
|-------------------------------------------- |----------:|----------:|----------:|----------:|
| YOLOv8 - CPU                                | 49.309 ms | 0.9628 ms | 1.7362 ms |   9.07 KB |
| YOLOv8 - GPU                                |  6.308 ms | 0.0116 ms | 0.0097 ms |   9.08 KB |
| YOLOv8 - GPU TensorRT FP32                  |  4.513 ms | 0.0098 ms | 0.0092 ms |   9.08 KB |
| YOLOv8 - GPU TensorRT FP16                  |  2.774 ms | 0.0032 ms | 0.0029 ms |   9.07 KB |
| YOLO11 - CPU                                | 43.759 ms | 0.8012 ms | 1.5244 ms |   9.06 KB |
| YOLO11 - GPU                                |  6.630 ms | 0.0100 ms | 0.0093 ms |   9.09 KB |
| YOLO11 - GPU TensorRT FP32                  |  4.635 ms | 0.0093 ms | 0.0087 ms |   9.08 KB |
| YOLO11 - GPU TensorRT FP16                  |  3.105 ms | 0.0163 ms | 0.0152 ms |   9.08 KB |
| YOLO26 - CPU                                | 43.360 ms | 0.8621 ms | 1.6403 ms |   4.54 KB |
| YOLO26 - GPU                                |  6.971 ms | 0.0109 ms | 0.0096 ms |   4.53 KB |
| YOLO26 - GPU TensorRT FP32                  |  4.631 ms | 0.0076 ms | 0.0071 ms |   4.55 KB |
| YOLO26 - GPU TensorRT FP16                  |  2.953 ms | 0.0096 ms | 0.0090 ms |   4.52 KB |
| YOLOv8 - GPU + CUDA Graph                   |  5.290 ms | 0.0086 ms | 0.0081 ms |   8.63 KB |
| YOLO11 - GPU + CUDA Graph                   |  5.548 ms | 0.0077 ms | 0.0072 ms |   8.48 KB |
| YOLO26 - GPU + CUDA Graph                   |  5.574 ms | 0.0158 ms | 0.0148 ms |   3.96 KB |
| YOLOv8 - GPU TensorRT FP32 + CUDA Graph     |  4.097 ms | 0.0075 ms | 0.0071 ms |    8.5 KB |
| YOLOv8 - GPU TensorRT FP16 + CUDA Graph     |  2.337 ms | 0.0056 ms | 0.0052 ms |   8.48 KB |
| YOLO11 - GPU TensorRT FP32 + CUDA Graph     |  4.026 ms | 0.0069 ms | 0.0061 ms |   8.49 KB |
| YOLO11 - GPU TensorRT FP16 + CUDA Graph     |  2.411 ms | 0.0069 ms | 0.0061 ms |   8.47 KB |
| YOLO26 - GPU TensorRT FP32 + CUDA Graph     |  4.033 ms | 0.0063 ms | 0.0049 ms |   3.95 KB |
| YOLO26 - GPU TensorRT FP16 + CUDA Graph     |  2.294 ms | 0.0114 ms | 0.0106 ms |   3.98 KB |

##### POSE ESTIMATION (input image size: 1280x720)
Image: [crosswalk.jpg](../assets/Media/crosswalk.jpg)

| Model                                       | Mean      | Error     | StdDev    | Allocated |
|-------------------------------------------- |----------:|----------:|----------:|----------:|
| YOLOv8 - CPU                                | 19.751 ms | 0.4472 ms | 1.2902 ms |  23.48 KB |
| YOLOv8 - GPU                                |  4.183 ms | 0.0109 ms | 0.0102 ms |  23.43 KB |
| YOLOv8 - GPU TensorRT FP32                  |  2.761 ms | 0.0051 ms | 0.0047 ms |  23.43 KB |
| YOLOv8 - GPU TensorRT FP16                  |  1.772 ms | 0.0113 ms | 0.0101 ms |  23.43 KB |
| YOLO11 - CPU                                | 15.475 ms | 0.3058 ms | 0.4386 ms |  22.47 KB |
| YOLO11 - GPU                                |  4.547 ms | 0.0110 ms | 0.0092 ms |  22.44 KB |
| YOLO11 - GPU TensorRT FP32                  |  2.901 ms | 0.0137 ms | 0.0128 ms |  22.43 KB |
| YOLO11 - GPU TensorRT FP16                  |  2.024 ms | 0.0099 ms | 0.0093 ms |  22.42 KB |
| YOLO26 - CPU                                | 15.809 ms | 0.3140 ms | 0.7341 ms |  11.35 KB |
| YOLO26 - GPU                                |  4.947 ms | 0.0438 ms | 0.0410 ms |  11.32 KB |
| YOLO26 - GPU TensorRT FP32                  |  2.997 ms | 0.0127 ms | 0.0106 ms |  11.35 KB |
| YOLO26 - GPU TensorRT FP16                  |  1.981 ms | 0.0132 ms | 0.0117 ms |  11.32 KB |
| YOLOv8 - GPU + CUDA Graph                   |  3.183 ms | 0.0443 ms | 0.0414 ms |  22.89 KB |
| YOLO11 - GPU + CUDA Graph                   |  3.353 ms | 0.0501 ms | 0.0469 ms |  21.86 KB |
| YOLO26 - GPU + CUDA Graph                   |  3.614 ms | 0.0535 ms | 0.0475 ms |  10.73 KB |
| YOLOv8 - GPU TensorRT FP32 + CUDA Graph     |  2.266 ms | 0.0284 ms | 0.0266 ms |  22.87 KB |
| YOLOv8 - GPU TensorRT FP16 + CUDA Graph     |  1.251 ms | 0.0022 ms | 0.0019 ms |  22.84 KB |
| YOLO11 - GPU TensorRT FP32 + CUDA Graph     |  2.211 ms | 0.0126 ms | 0.0112 ms |  21.85 KB |
| YOLO11 - GPU TensorRT FP16 + CUDA Graph     |  1.295 ms | 0.0054 ms | 0.0050 ms |  21.83 KB |
| YOLO26 - GPU TensorRT FP32 + CUDA Graph     |  2.246 ms | 0.0322 ms | 0.0302 ms |  10.74 KB |
| YOLO26 - GPU TensorRT FP16 + CUDA Graph     |  1.253 ms | 0.0027 ms | 0.0025 ms |  10.72 KB |

##### SEGMENTATION (input image size: 1280x853)
Image: [people.jpg](../assets/Media/people.jpg)

| Model                                       | Mean      | Error     | StdDev    | Gen0   | Allocated |
|-------------------------------------------- |----------:|----------:|----------:|-------:|----------:|
| YOLOv8 - CPU                                | 43.840 ms | 0.2639 ms | 0.2468 ms |      - | 489.38 KB |
| YOLOv8 - GPU                                |  5.480 ms | 0.0156 ms | 0.0139 ms | 7.8125 | 489.47 KB |
| YOLOv8 - GPU TensorRT FP32                  |  3.987 ms | 0.0111 ms | 0.0104 ms | 7.8125 |  489.3 KB |
| YOLOv8 - GPU TensorRT FP16                  |  2.888 ms | 0.0180 ms | 0.0168 ms | 7.8125 | 491.73 KB |
| YOLO11 - CPU                                | 37.819 ms | 0.1777 ms | 0.1576 ms |      - | 458.21 KB |
| YOLO11 - GPU                                |  5.825 ms | 0.0146 ms | 0.0129 ms | 7.8125 | 457.08 KB |
| YOLO11 - GPU TensorRT FP32                  |  4.091 ms | 0.0258 ms | 0.0242 ms | 7.8125 | 457.12 KB |
| YOLO11 - GPU TensorRT FP16                  |  3.050 ms | 0.0158 ms | 0.0148 ms | 7.8125 | 456.62 KB |
| YOLO26 - CPU                                | 38.847 ms | 0.2369 ms | 0.2216 ms |      - | 523.96 KB |
| YOLO26 - GPU                                |  6.182 ms | 0.0168 ms | 0.0157 ms |      - | 527.99 KB |
| YOLO26 - GPU TensorRT FP32                  |  4.102 ms | 0.0208 ms | 0.0174 ms | 7.8125 | 528.21 KB |
| YOLO26 - GPU TensorRT FP16                  |  2.972 ms | 0.0173 ms | 0.0162 ms | 7.8125 | 527.69 KB |
| YOLOv8 - GPU + CUDA Graph                   |  4.455 ms | 0.0108 ms | 0.0101 ms | 7.8125 | 488.73 KB |
| YOLO11 - GPU + CUDA Graph                   |  4.580 ms | 0.0084 ms | 0.0075 ms | 7.8125 | 456.37 KB |
| YOLO26 - GPU + CUDA Graph                   |  4.790 ms | 0.0290 ms | 0.0271 ms | 7.8125 | 527.28 KB |
| YOLOv8 - GPU TensorRT FP32 + CUDA Graph     |  3.509 ms | 0.0119 ms | 0.0105 ms | 7.8125 | 488.54 KB |
| YOLOv8 - GPU TensorRT FP16 + CUDA Graph     |  2.428 ms | 0.0205 ms | 0.0192 ms | 7.8125 |    491 KB |
| YOLO11 - GPU TensorRT FP32 + CUDA Graph     |  3.512 ms | 0.0430 ms | 0.0381 ms | 7.8125 | 456.37 KB |
| YOLO11 - GPU TensorRT FP16 + CUDA Graph     |  2.336 ms | 0.0084 ms | 0.0079 ms | 7.8125 | 455.87 KB |
| YOLO26 - GPU TensorRT FP32 + CUDA Graph     |  3.417 ms | 0.0127 ms | 0.0119 ms | 7.8125 | 527.60 KB |
| YOLO26 - GPU TensorRT FP16 + CUDA Graph     |  2.139 ms | 0.0138 ms | 0.0129 ms | 7.8125 | 526.97 KB |
