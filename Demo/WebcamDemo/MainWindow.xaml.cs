// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

using OpenCvSharp;
using SkiaSharp;
using SkiaSharp.Views.Desktop;
using System.Diagnostics;
using System.Windows;
using System.Windows.Threading;
using YoloDotNet;
using YoloDotNet.Enums;
using YoloDotNet.ExecutionProvider.Cuda;
using YoloDotNet.Extensions;
using YoloDotNet.Models;
using YoloDotNet.Test.Common;
using YoloDotNet.Trackers;
using Window = System.Windows.Window;

namespace WebcamDemo
{
    /// <summary>
    /// Demonstrates real-time object detection and tracking from a webcam using YoloDotNet and EmguCV.
    /// 
    /// This demo captures frames directly from the webcam, performs object detection using YOLO models, 
    /// and optionally applies multi-object tracking (SORT). Detected objects are drawn on the frames with 
    /// bounding boxes, labels, confidence scores, and tracked tails.
    /// 
    /// It showcases:
    /// - Model initialization with configurable hardware acceleration (CUDA, with optional TensorRT integration) and preprocessing options
    /// - Real-time object detection on live webcam input using YoloDotNet
    /// - Optional class label filtering (e.g., detecting only persons)
    /// - Optional multi-object tracking across frames using the SORT tracker
    /// - Direct rendering of detection and tracking results on the live video feed
    /// - Frame processing time reporting for performance monitoring
    /// 
    /// Example webcam sources:
    /// - Default webcam device (index 0)
    /// - Additional devices by index (e.g., 1 for a secondary camera)
    /// 
    /// Execution providers:
    /// - CpuExecutionProvider: runs inference on CPU, universally supported but slower.
    /// - CudaExecutionProvider: executes inference on an NVIDIA GPU using CUDA for accelerated performance.
    ///   Optionally integrates with TensorRT for further optimization, supporting FP32, FP16, and INT8 precision modes.
    ///   This delivers significant speed improvements on compatible GPUs.
    /// 
    /// Important notes:
    /// - Choose the execution provider based on your hardware and performance requirements.
    /// - If using CUDA with TensorRT enabled, ensure your environment has a compatible CUDA, cuDNN, and TensorRT setup.
    /// - For detailed setup instructions and examples, see the README:
    ///   https://github.com/NickSwardh/YoloDotNet
    /// </summary>
    public partial class MainWindow : Window
    {
        #region Private fields
        private readonly Yolo _yolo = default!;
        private readonly SortTracker _sortTracker = default!;
        private SKBitmap _currentFrame = default!;
        private Dispatcher _dispatcher = default!;
        private CancellationTokenSource _cts;
        private bool _runDetection = false;
        private SKRect _rect;
        private Stopwatch _stopwatch = default!;

        private bool _isTrackingEnabled;
        private bool _isFilteringEnabled;
        private double _confidenceThreshold;
        #endregion

        #region Constants
        const int WEBCAM_WIDTH = 1280;
        const int WEBCAM_HEIGHT = 720;
        const int FPS = 30;
        #endregion

        public MainWindow()
        {
            InitializeComponent();

            // Initialize stopwatch for simple measuring of frame processing time
            _stopwatch = new Stopwatch();

            // (Optional) Create a new SortTracker instance with configurable parameters:
            // - costThreshold: matching cost threshold for assigning detections to tracks (lower = stricter matching).
            // - maxAge: number of frames to keep unmatched tracks before removal.
            // - tailLength: length of the track history for visualization or analysis.
            // Note: There is no one-size-fits-all setting; these parameters often require some tinkering to find the best balance for your specific use case.
            _sortTracker = new SortTracker(costThreshold: 0.5f, maxAge: 5, tailLength: 30);

            // Initialize YoloDotNet.
            // YoloOptions configures the model, hardware settings, and image processing behavior.
            _yolo = new Yolo(new YoloOptions
            {
                // Select execution provider (determines how and where inference is executed).
                // Available execution providers:
                // 
                //   - CpuExecutionProvider
                //     Runs inference entirely on the CPU. Universally supported on all hardware.
                //
                //   - CudaExecutionProvider
                //     Executes inference on an NVIDIA GPU using CUDA for accelerated performance.  
                //     Optionally integrates with TensorRT for further optimization, supporting FP32, FP16,  
                //     and INT8 precision modes. This delivers significant speed improvements on compatible GPUs.  
                //     See the TensorRT demo and documentation for detailed configuration and best practices.
                //
                //   - OpenVinoExecutionProvider
                //     Runs inference using Intel's OpenVINO toolkit for optimized performance on Intel hardware.
                //
                //   - CoreMLExecutionProvider
                //     Executes inference using Apple's CoreML framework for efficient performance on macOS and iOS devices.
                //
                //   Important:  
                //     - Choose the provider that matches your available hardware and performance requirements.  
                //     - If using CUDA with TensorRT enabled, ensure your environment has a compatible CUDA, cuDNN, and TensorRT setup.
                //     - For detailed setup instructions and examples, see the README:
                //
                //   More information about execution providers and setup instructions can be found in the README:
                //   https://github.com/NickSwardh/YoloDotNet

                ExecutionProvider = new CudaExecutionProvider(

                    // Path or byte[] of the ONNX model to load.
                    model: SharedConfig.GetTestModelV26(ModelType.ObjectDetection),

                    // GPU device Id to use for inference. -1 = CPU, 0+ = GPU device Id.
                    gpuId: 0),

                // Resize mode applied before inference. Proportional maintains the aspect ratio (adds padding if needed),
                // while Stretch resizes the image to fit the target size without preserving the aspect ratio.
                // Set this accordingly, as it directly impacts the inference results.
                ImageResize = ImageResize.Proportional,

                // Sampling options for resizing; affects inference speed and quality.
                // For examples of other sampling options, see benchmarks: https://github.com/NickSwardh/YoloDotNet/tree/master/test/YoloDotNet.Benchmarks
                SamplingOptions = new(SKFilterMode.Nearest, SKMipmapMode.None) // YoloDotNet default

            });

            _dispatcher = Dispatcher.CurrentDispatcher;

            _cts = new CancellationTokenSource();

            _currentFrame = new SKBitmap(WEBCAM_WIDTH, WEBCAM_HEIGHT, SKColorType.Bgra8888, SKAlphaType.Premul);
            _rect = new SKRect(0, 0, WEBCAM_WIDTH, bottom: WEBCAM_HEIGHT);

            // Start the webcam capture on a background thread
            Task.Run(() => WebcamAsync(), _cts.Token);
        }

        private async Task WebcamAsync()
        {
            // Initialize the webcam
            using var capture = new VideoCapture(0, VideoCaptureAPIs.DSHOW);

            capture.Set(VideoCaptureProperties.Fps, FPS);
            capture.Set(VideoCaptureProperties.FrameWidth, WEBCAM_WIDTH);
            capture.Set(VideoCaptureProperties.FrameHeight, WEBCAM_HEIGHT);

            // If the camera supports MJPEG, it's much cheaper on CPU than uncompressed frames.
            capture.Set(VideoCaptureProperties.FourCC, VideoWriter.FourCC('M', 'J', 'P', 'G'));

            using var mat = new Mat();
            using var bgraMat = new Mat();

            var drawingOptions = new DetectionDrawingOptions
            {
                DrawLabels = false
            };

            // Continuously capture frames from the webcam until cancellation is requested, eg. when closing the window.
            while (_cts.IsCancellationRequested is false)
            {
                // Capture the current frame from the webcam
                capture.Read(mat);
                
                // Convert the frame to BGRA color space
                Cv2.CvtColor(mat, bgraMat, ColorConversionCodes.BGR2BGRA);

                _currentFrame.SetPixels(bgraMat.Data);

                if (_runDetection)
                {
                    _stopwatch.Restart();

                    // Run object detection on the current frame
                    var results = _yolo.RunObjectDetection(_currentFrame, _confidenceThreshold, iou: 0.5);

                    _stopwatch.Stop();

                    // Optionally filter results to only include specific class labels
                    if (_isFilteringEnabled)
                        results = results.FilterLabels(["person", "cat", "dog"]);

                    // Optionally track objects using the SortTracker
                    if (_isTrackingEnabled)
                        results.Track(_sortTracker);

                    // Draw detection and tracking results on the current frame
                    _currentFrame.Draw(results);
                }

                // Update the SKElement on the UI thread
                await _dispatcher.InvokeAsync(() =>
                {
                    WebCamFrame.InvalidateVisual(); // Notify SKiaSharp to update the frame.

                    // Display processing time and max fps
                    if (_runDetection)
                    {
                        var milliseconds = _stopwatch.Elapsed.TotalMilliseconds;
                        var yoloFps = 1000.0 / milliseconds;

                        FrameProcess.Text = $"Processed Frame: {milliseconds:F1}ms ({yoloFps:F1} fps)";
                    }
                },
                DispatcherPriority.Render, // Ensure the UI updates accordingly.
                _cts.Token);
            }
        }

        private void UpdateWebcamFrame(object sender, SKPaintSurfaceEventArgs e)
        {
            using var canvas = e.Surface.Canvas;
            canvas.DrawBitmap(_currentFrame, _rect);
        }

        private void StartClick(object sender, RoutedEventArgs e)
            => _runDetection = true;

        private void StopClick(object sender, RoutedEventArgs e)
            => _runDetection = false;

        private void EnableFiltering_Checked(object sender, RoutedEventArgs e)
            => _isFilteringEnabled = true;

        private void EnableFiltering_Unchecked(object sender, RoutedEventArgs e)
            => _isFilteringEnabled = false;

        private void EnableTracking_Checked(object sender, RoutedEventArgs e)
            => _isTrackingEnabled = true;

        private void EnableTracking_Unchecked(object sender, RoutedEventArgs e)
            => _isTrackingEnabled = false;

        private void WindowClosing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            _cts.Cancel();

            Thread.Sleep(500); // Give some time for the webcam task to end gracefully.

            _runDetection = false;
            _yolo?.Dispose();
            _currentFrame?.Dispose();
        }

        private void ConfidenceTreshold_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            _confidenceThreshold = e.NewValue;
        }
    }
}