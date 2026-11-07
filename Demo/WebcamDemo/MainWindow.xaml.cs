// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

using System.ComponentModel;
using System.Diagnostics;
using System.Windows;
using System.Windows.Threading;
using OpenCvSharp;
using SkiaSharp;
using SkiaSharp.Views.Desktop;
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
    /// Real-time object detection, with optional tracking and label filtering, on a webcam feed.
    /// See README.md in this folder for how the demo works and what you can change.
    /// </summary>
    public partial class MainWindow : Window
    {
        private const int CAMERA_INDEX = 0;
        private const int WEBCAM_WIDTH = 1280;
        private const int WEBCAM_HEIGHT = 720;
        private const int FPS = 30;

        // Labels kept when "Enable Filtering" is checked
        private static readonly HashSet<string> _filteredLabels = ["person", "cat", "dog"];

        private readonly Yolo _yolo;
        private readonly SortTracker _sortTracker;
        private readonly SKBitmap _currentFrame;
        private readonly SKRect _frameRect;
        private readonly CancellationTokenSource _cts = new();
        private readonly Stopwatch _stopwatch = new();

        private bool _runDetection;
        private bool _isTrackingEnabled;
        private bool _isFilteringEnabled;
        private double _confidenceThreshold;

        public MainWindow()
        {
            InitializeComponent();
            _confidenceThreshold = ConfidenceThreshold.Value;

            _yolo = new Yolo(new YoloOptions
            {
                ExecutionProvider = new CudaExecutionProvider(SharedConfig.GetTestModelV26(ModelType.ObjectDetection))
            });

            _sortTracker = new SortTracker(maxAge: 5);

            // Webcam frames are fully opaque (BGR2BGRA sets every alpha to 255), so label them Opaque.
            _currentFrame = new SKBitmap(WEBCAM_WIDTH, WEBCAM_HEIGHT, SKColorType.Bgra8888, SKAlphaType.Opaque);
            _frameRect = new SKRect(0, 0, WEBCAM_WIDTH, WEBCAM_HEIGHT);

            // Capture and process frames on a background thread
            Task.Run(() => WebcamAsync(), _cts.Token);
        }

        private async Task WebcamAsync()
        {
            using var capture = new VideoCapture(CAMERA_INDEX, VideoCaptureAPIs.DSHOW);
            capture.Set(VideoCaptureProperties.Fps, FPS);
            capture.Set(VideoCaptureProperties.FrameWidth, WEBCAM_WIDTH);
            capture.Set(VideoCaptureProperties.FrameHeight, WEBCAM_HEIGHT);

            // MJPEG, if the camera supports it, is much cheaper on the CPU than uncompressed frames.
            capture.Set(VideoCaptureProperties.FourCC, VideoWriter.FourCC('M', 'J', 'P', 'G'));

            using var mat = new Mat();
            using var bgraMat = new Mat();

            // Runs until the window closes
            while (_cts.IsCancellationRequested is false)
            {
                capture.Read(mat);
                Cv2.CvtColor(mat, bgraMat, ColorConversionCodes.BGR2BGRA);
                _currentFrame.SetPixels(bgraMat.Data);

                if (_runDetection)
                {
                    _stopwatch.Restart();
                    var results = _yolo.RunObjectDetection(_currentFrame, _confidenceThreshold, iou: 0.5);
                    _stopwatch.Stop();

                    if (_isFilteringEnabled)
                        results = results.FilterLabels(_filteredLabels);

                    if (_isTrackingEnabled)
                        results.Track(_sortTracker);

                    _currentFrame.Draw(results);
                }

                await Dispatcher.InvokeAsync(() =>
                {
                    WebCamFrame.InvalidateVisual(); // Repaints the frame through UpdateWebcamFrame

                    if (_runDetection)
                    {
                        var milliseconds = _stopwatch.Elapsed.TotalMilliseconds;
                        FrameProcess.Text = $"Processed Frame: {milliseconds:F1}ms ({1000.0 / milliseconds:F1} fps)";
                    }
                },
                DispatcherPriority.Render,
                _cts.Token);
            }
        }

        private void UpdateWebcamFrame(object sender, SKPaintSurfaceEventArgs e)
        {
            using var canvas = e.Surface.Canvas;
            canvas.DrawBitmap(_currentFrame, _frameRect);
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

        private void ConfidenceThreshold_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
            => _confidenceThreshold = e.NewValue;

        private void WindowClosing(object sender, CancelEventArgs e)
        {
            _cts.Cancel();
            Thread.Sleep(500); // Give the capture loop time to finish

            _runDetection = false;
            _yolo.Dispose();
            _currentFrame.Dispose();
        }
    }
}
