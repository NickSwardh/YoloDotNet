// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2024-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

using Emgu.CV;
using Emgu.CV.CvEnum;
using SkiaSharp;
using SkiaSharp.Views.Desktop;
using System.Diagnostics;
using System.Windows;
using System.Windows.Threading;
using YoloDotNet;
using YoloDotNet.Enums;
using YoloDotNet.Extensions;
using YoloDotNet.Models;
using YoloDotNet.Test.Common;
using YoloDotNet.Trackers;

namespace WebcamDemo
{
    /// <summary>
    /// Demonstrates real-time object detection and tracking from a webcam using the YoloDotNet and EmguCV.
    /// 
    /// This demo captures frames directly from the webcam, performs object detection using YOLO models, 
    /// and optionally applies multi-object tracking (SORT). Detected objects are drawn on the frames with 
    /// bounding boxes, labels, confidence scores, and tracked tails.
    /// 
    /// It showcases:
    /// - Model initialization with configurable hardware acceleration (CUDA) and preprocessing options
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
    /// Note:
    /// - CUDA acceleration is enabled for faster inference.
    /// - The demo updates the WPF UI in real time with processed frames and performance metrics.
    /// </summary>
    public partial class MainWindow : Window
    {
        #region Fields

        private readonly Yolo _yolo = default!;
        private readonly SortTracker _sortTracker = default!;
        private SKBitmap _currentFrame = default!;
        private Dispatcher _dispatcher = default!;

        private bool _runDetection = false;
        private SKRect _rect;
        private Stopwatch _stopwatch = default!;

        private SKImageInfo _imageInfo = default!;
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

            // Instantiate yolo 11
            _yolo = new Yolo(new YoloOptions()
            {
                OnnxModel = SharedConfig.GetTestModelV11(ModelType.ObjectDetection),
                Cuda = true,
                PrimeGpu = false,
                ImageResize = ImageResize.Proportional,
                SamplingOptions = new(SKFilterMode.Nearest, SKMipmapMode.None) // YoloDotNet default
            });

            _dispatcher = Dispatcher.CurrentDispatcher;

            _currentFrame = new SKBitmap(WEBCAM_WIDTH, WEBCAM_HEIGHT);
            _rect = new SKRect(0, 0, WEBCAM_WIDTH, bottom: WEBCAM_HEIGHT);
            _imageInfo = new SKImageInfo(WEBCAM_WIDTH, WEBCAM_HEIGHT, SKColorType.Bgra8888, SKAlphaType.Premul);

            // Start the webcam capture on a background thread
            Task.Run(() => WebcamAsync());
        }

        private async Task WebcamAsync()
        {
            // Initialize the webcam
            using var capture = new VideoCapture(0, VideoCapture.API.DShow);

            capture.Set(CapProp.Fps, FPS);
            capture.Set(CapProp.FrameWidth, WEBCAM_WIDTH);
            capture.Set(CapProp.FrameHeight, WEBCAM_HEIGHT);

            using var mat = new Mat();
            using var bgraMat = new Mat();

            while (true)
            {
                // Capture the current frame from the webcam
                capture.Read(mat);

                // Convert the frame to BGRA color space
                CvInvoke.CvtColor(mat, bgraMat, ColorConversion.Bgr2Bgra);

                // Create an SKBitmap from the BGRA Mat for processing
                using var frame = SKImage.FromPixels(_imageInfo, bgraMat.DataPointer);
                _currentFrame = SKBitmap.FromImage(frame);

                _stopwatch.Restart();

                if (_runDetection)
                {
                    // Run object detection on the current frame
                    var results = _yolo.RunObjectDetection(_currentFrame, _confidenceThreshold, iou: 0.7);

                    if (_isFilteringEnabled)
                        results = results.FilterLabels(["person", "cat", "plane"]);  // Optionally filter results to include only specific classes (e.g., "person", "cat", "plan")

                    if (_isTrackingEnabled)
                        results = results.Track(_sortTracker); // Optionally track objects using the SortTracker

                    // Draw detection and tracking results on the current frame
                    _currentFrame.Draw(results);
                }

                _stopwatch.Stop();

                // Update GUI
                await _dispatcher.InvokeAsync(() =>
                {
                    WebCamFrame.InvalidateVisual(); // Notify SKiaSharp to update the frame in the GUI.
                    FrameProcess.Text = $"Processed Frame (ms): {_stopwatch.ElapsedMilliseconds}";
                });
            }
        }

        private void UpdateWebcamFrame(object sender, SKPaintSurfaceEventArgs e)
        {
            using var canvas = e.Surface.Canvas;
            canvas.DrawBitmap(_currentFrame, _rect);
            canvas.Flush();
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