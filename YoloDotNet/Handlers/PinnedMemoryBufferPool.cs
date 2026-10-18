// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Handlers
{
    /// <summary>
    /// A pool for managing reusable pinned memory buffers backed by SKBitmap instances.
    /// Helps reduce GC pressure and allocation cost in high-frequency image processing scenarios.
    /// </summary>
    public class PinnedMemoryBufferPool : IDisposable
    {
        // Internal thread-safe pool of reusable pinned memory buffers
        internal readonly ConcurrentBag<PinnedMemoryBuffer> _pool = [];

        // The image format/dimensions used to allocate each SKBitmap
        private readonly SKImageInfo _imageInfo;

        /// <summary>
        /// Initializes the buffer pool with a specified image layout and pre-allocates a number of buffers.
        /// </summary>
        public PinnedMemoryBufferPool(SKImageInfo skInfo, int initialSize = 2)
        {
            _imageInfo = skInfo;

            for (int i = 0; i < initialSize; i++)
                _pool.Add(new PinnedMemoryBuffer(_imageInfo));
        }

        /// <summary>
        /// Retrieves a buffer from the pool, or creates a new one if the pool is empty.
        /// The buffer is filled with the letterbox padding colour before it is handed out.
        /// </summary>
        public PinnedMemoryBuffer Rent()
        {
            // Pool exhausted — create a new buffer as a fallback
            if (!_pool.TryTake(out var buffer))
                buffer = new PinnedMemoryBuffer(_imageInfo);

            // IMPORTANT: Clear on rent (not on return) so freshly allocated buffers, which start zeroed (black),
            // get the same padding as reused ones. This also avoids leaking old frame data into subsequent frames.
            buffer.TargetBitmap.Erase(ImageConfig.LetterboxPaddingColor);

            return buffer;
        }

        /// <summary>
        /// Returns a used buffer back to the pool.
        /// </summary>
        /// <param name="buffer">The buffer to be returned and reused.</param>
        public void Return(PinnedMemoryBuffer buffer)
            => _pool.Add(buffer);

        /// <summary>
        /// Releases all resources used by the pool.
        /// </summary>
        public void Dispose()
        {
            while (_pool.TryTake(out var buffer))
                buffer?.Dispose();

            GC.SuppressFinalize(this);
        }
    }
}
