namespace YoloDotNet.Handlers
{
    public class PinnedMemoryBufferPool : IDisposable
    {
        internal readonly ConcurrentBag<PinnedMemoryBuffer> _pool = [];
        private readonly SKImageInfo _imageInfo;

        public PinnedMemoryBufferPool(SKImageInfo skInfo, int initialSize = 20)
        {
            _imageInfo = skInfo;

            for (int i = 0; i < initialSize; i++)
                _pool.Add(new PinnedMemoryBuffer(_imageInfo));
        }

        public PinnedMemoryBuffer Rent()
        {
            if (_pool.TryTake(out var buffer))
                return buffer;

            return new PinnedMemoryBuffer(_imageInfo); // fallback
        }

        public void Return(PinnedMemoryBuffer buffer)
        {
            //buffer.TargetBitmap.Erase(SKColors.Empty);
            _pool.Add(buffer);
        }

        public void Dispose()
        {
            while (_pool.TryTake(out var buffer))
                buffer?.Dispose();

            GC.SuppressFinalize(this);
        }
    }
}
