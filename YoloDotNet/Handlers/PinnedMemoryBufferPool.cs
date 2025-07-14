// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
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
        public PinnedMemoryBufferPool(SKImageInfo skInfo, int initialSize = 60)
        {
            _imageInfo = skInfo;

            for (int i = 0; i < initialSize; i++)
                _pool.Add(new PinnedMemoryBuffer(_imageInfo));
        }

        /// <summary>
        /// Retrieves a buffer from the pool, or creates a new one if the pool is empty.
        /// </summary>
        public PinnedMemoryBuffer Rent()
        {
            if (_pool.TryTake(out var buffer))
                return buffer;

            // Pool exhausted — create a new buffer as a fallback
            return new PinnedMemoryBuffer(_imageInfo); // fallback
        }

        /// <summary>
        /// Returns a used buffer back to the pool after clearing its contents.
        /// </summary>
        /// <param name="buffer">The buffer to be returned and reused.</param>
        public void Return(PinnedMemoryBuffer buffer)
        {
            // IMPORTANT: Clear the bitmap before reuse to prevent visual artifacts.
            // This avoids leaking old frame data into subsequent frames.
            // Using SKColors.Empty fills with transparent black (0,0,0,0).
            buffer.TargetBitmap.Erase(SKColors.Empty);

            _pool.Add(buffer);
        }

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
