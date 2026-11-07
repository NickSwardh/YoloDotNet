// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Video.Services
{
    /// <summary>
    /// Writes saved images to disk on a background thread, so callers such as video frame handlers don't wait for disk I/O.
    /// Images are encoded on the caller's thread, and every queued image is written before the process exits.
    /// </summary>
    internal static class FrameSaveService
    {
        private static readonly BlockingCollection<(byte[] ImageBytes, string FileName)> _frameQueue = new(100);
        private static readonly Task _writerTask;

        static FrameSaveService()
        {
            _writerTask = Task.Factory.StartNew(WriteQueuedImages, TaskCreationOptions.LongRunning);

            // Write the images that are still queued before the process exits
            AppDomain.CurrentDomain.ProcessExit += (_, _) =>
            {
                _frameQueue.CompleteAdding();
                _writerTask.Wait();
            };
        }

        /// <summary>
        /// Encodes the image and queues it to be written to <paramref name="fileName"/>.
        /// </summary>
        public static void AddToQueue(SKBitmap image,
            string fileName,
            SKEncodedImageFormat format = SKEncodedImageFormat.Jpeg,
            int quality = 100)
        {
            using var memoryStream = new MemoryStream();
            image.Encode(memoryStream, format, quality);

            _frameQueue.Add((memoryStream.ToArray(), fileName));
        }

        /// <summary>
        /// Encodes the image and queues it to be written to <paramref name="fileName"/>.
        /// </summary>
        public static void AddToQueue(SKImage image,
            string fileName,
            SKEncodedImageFormat format = SKEncodedImageFormat.Jpeg,
            int quality = 100)
        {
            using var imageData = image.Encode(format, quality);

            _frameQueue.Add((imageData.ToArray(), fileName));
        }

        private static void WriteQueuedImages()
        {
            foreach (var (imageBytes, fileName) in _frameQueue.GetConsumingEnumerable())
            {
                try
                {
                    // Replaces an existing file completely, so no bytes of a larger old file remain
                    File.WriteAllBytes(fileName, imageBytes);
                }
                catch (Exception)
                {
                    // A failed write, such as a missing folder or no permission, must not stop the images queued after it
                }
            }
        }
    }
}
