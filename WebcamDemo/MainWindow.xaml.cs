using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Util;
using SkiaSharp;
using SkiaSharp.Views.Desktop;
using System.Windows;
using System.Windows.Threading;
using YoloDotNet;
using YoloDotNet.Enums;
using YoloDotNet.Extensions;
using YoloDotNet.Models;
using YoloDotNet.Test.Common;

namespace WebcamDemo
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        #region Fields

        private readonly Yolo _yolo = default!;
        private SKImage _currentFrame = default!;
        private Dispatcher _dispatcher = default!;

        private bool _runDetection = false;

        private SKRect _rect;

        #endregion

        #region Constants

        const int WEBCAM_WIDTH = 1080;
        const int WEBCAM_HEIGHT = 608;
        const int FPS = 30;
        const string FRAME_FORMAT_EXTENSION = ".png";

        #endregion

        public MainWindow()
        {
            InitializeComponent();

            // Instantiate yolo
            _yolo = new Yolo(new YoloOptions()
            {
                OnnxModel = SharedConfig.GetTestModelV11(ModelType.ObjectDetection),
                ModelType = ModelType.ObjectDetection,
                Cuda = true,
                PrimeGpu = false
            });

            _dispatcher = Dispatcher.CurrentDispatcher;

            _currentFrame = SKImage.FromBitmap(new SKBitmap(WEBCAM_WIDTH, WEBCAM_HEIGHT));
            _rect = new SKRect(0, 0, WEBCAM_WIDTH, bottom: WEBCAM_HEIGHT);

            // Start webcam on a separate thread
            Task.Run(() => WebcamAsync());
        }

        private async Task WebcamAsync()
        {
            // Configure webcam
            using var capture = new VideoCapture(0, VideoCapture.API.DShow);
            capture.Set(CapProp.FrameCount, FPS);
            capture.Set(CapProp.FrameWidth, WEBCAM_WIDTH);
            capture.Set(CapProp.FrameHeight, WEBCAM_HEIGHT);

            using var mat = new Mat();
            using var buffer = new VectorOfByte();

            while (true)
            {
                // Capture current frame from webcam
                capture.Read(mat);

                // Encode mat to a valid image format and to a buffer
                CvInvoke.Imencode(FRAME_FORMAT_EXTENSION, mat, buffer);

                // "Rewind" buffer
                buffer.Position = 0;

                // Read buffer to an SKImage
                _currentFrame = SKImage.FromEncodedData(buffer);

                // Clean up
                buffer.Clear();

                if (_runDetection)
                {
                    // Run inference on frame
                    var results = _yolo.RunObjectDetection(_currentFrame);

                    // Draw results
                    _currentFrame = _currentFrame.Draw(results);
                }

                // Update GUI
                await _dispatcher.InvokeAsync(() => WebCamFrame.InvalidateVisual());
            }
        }

        private void UpdateWebcamFrame(object sender, SKPaintSurfaceEventArgs e)
        {
            using var canvas = e.Surface.Canvas;
            canvas.DrawImage(_currentFrame, _rect);
            canvas.Flush();
        }

        private void StartClick(object sender, RoutedEventArgs e)
            => _runDetection = true;

        private void StopClick(object sender, RoutedEventArgs e)
            => _runDetection = false;

        private void WindowClosing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            _yolo?.Dispose();
            _currentFrame?.Dispose();
        }
    }
}