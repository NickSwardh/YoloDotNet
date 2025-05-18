namespace YoloDotNet.Video.Services
{
    internal static class FrameSaveService
    {
        private static readonly BlockingCollection<(byte[] frameBytes, string fileName)> _frameQueue;
        private static CancellationTokenSource _cancellationTokenSource = default!;
        private static Task _backgroundTask = default!;

        private static MemoryStream _memoryStream = default!;
        private static bool _isRunning;

        static FrameSaveService()
        {
            _memoryStream = new MemoryStream();
            _frameQueue = [];
        }

        /// <summary>
        /// Add SKBitmap to queue
        /// </summary>
        /// <param name="image"></param>
        /// <param name="fileName"></param>
        /// <param name="format"></param>
        /// <param name="quality"></param>
        public static void AddToQueue(SKBitmap image,
            string fileName,
            SKEncodedImageFormat format = SKEncodedImageFormat.Jpeg,
            int quality = 100)
        {
            _memoryStream.Position = 0;

            image.Encode(_memoryStream, format, quality);
            byte[] encodedBytes = _memoryStream.ToArray();

            _frameQueue.Add((encodedBytes, fileName));
        }

        /// <summary>
        /// Add SKImage to queue
        /// </summary>
        /// <param name="image"></param>
        /// <param name="fileName"></param>
        /// <param name="format"></param>
        /// <param name="quality"></param>
        public static void AddToQueue(SKImage image,
            string fileName,
            SKEncodedImageFormat format = SKEncodedImageFormat.Jpeg,
            int quality = 100)
        {
            _memoryStream.Position = 0;

            using var imageData = image.Encode(format, quality);
            imageData.SaveTo(_memoryStream);

            byte[] encodedBytes = _memoryStream.ToArray();

            _frameQueue.Add((encodedBytes, fileName));
        }

        public static void Start()
        {
            if (_isRunning is true)
                return;

            _isRunning = true;

            _cancellationTokenSource = new();
            _backgroundTask = Task.Run(ProcessQueue, _cancellationTokenSource.Token);
        }

        private static void Stop()
        {
            _cancellationTokenSource?.Cancel();
            _frameQueue.CompleteAdding();
            _backgroundTask?.Wait();
        }

        private static void ProcessQueue()
        {
            try
            {
                foreach (var (imageBytes, fileName) in _frameQueue.GetConsumingEnumerable(_cancellationTokenSource.Token))
                {
                    using var fileStream = new FileStream(
                        fileName,
                        FileMode.OpenOrCreate,
                        FileAccess.ReadWrite,
                        FileShare.ReadWrite,
                        4096,
                        true);

                    fileStream.Write(imageBytes, 0, imageBytes.Length);
                }
            }
            catch (OperationCanceledException)
            {
                // Exit gracefully.
            }
            catch (Exception)
            {
                // TODO: Handle any issues with saving the image, like IO errors, permissions, etc.
            }
        }

        public static void DisposeStaticFrameQueue()
        {
            Stop();

            _frameQueue?.Dispose();
            _cancellationTokenSource?.Dispose();
            _backgroundTask?.Dispose();
        }
    }
}
