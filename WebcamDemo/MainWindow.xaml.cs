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
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        #region Fields

        private readonly Yolo _yolo = default!;
        private readonly SortTrack _sortTracker = default!;
        private SKBitmap _currentFrame = default!;
        private Dispatcher _dispatcher = default!;

        private bool _runDetection = false;
        private SKRect _rect;
        private Stopwatch _stopwatch = default!;

        #endregion

        #region Constants

        const int WEBCAM_WIDTH = 1080;
        const int WEBCAM_HEIGHT = 720;
        const int FPS = 30;

        #endregion

        public MainWindow()
        {
            InitializeComponent();

            _stopwatch = new Stopwatch();
            _sortTracker = new SortTrack();

            // Instantiate yolo 11
            _yolo = new Yolo(new YoloOptions()
            {
                OnnxModel = SharedConfig.GetTestModelV11(ModelType.ObjectDetection),
                ModelType = ModelType.ObjectDetection,
                Cuda = true,
                PrimeGpu = false
            });

            _dispatcher = Dispatcher.CurrentDispatcher;

            _currentFrame = new SKBitmap(WEBCAM_WIDTH, WEBCAM_HEIGHT);
            _rect = new SKRect(0, 0, WEBCAM_WIDTH, bottom: WEBCAM_HEIGHT);

            // Start webcam on a separate thread
            Task.Run(() => WebcamAsync());
        }

        private async Task WebcamAsync()
        {
            // Configure webcam
            using var capture = new VideoCapture(0, VideoCapture.API.DShow);

            capture.Set(CapProp.Fps, FPS);
            capture.Set(CapProp.FrameWidth, WEBCAM_WIDTH);
            capture.Set(CapProp.FrameHeight, WEBCAM_HEIGHT);

            using var mat = new Mat();

            while (true)
            {
                // Capture current frame from webcam
                capture.Read(mat);

                _stopwatch.Restart();
                _currentFrame = mat.ToBitmap().ToSKBitmap();

                if (_runDetection)
                {
                    // Run inference on frame
                    var results = _yolo.RunObjectDetection(_currentFrame, 0.25)
                        .FilterLabels(["person"])
                        .Track(_sortTracker);

                    // Update _currentFrame with drawn results
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

        private void WindowClosing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            _runDetection = false;
            _yolo?.Dispose();
            _currentFrame?.Dispose();
        }
    }
}