namespace YoloDotNet.Handlers
{
    public class PinnedMemoryBuffer
    {
        public readonly SKImageInfo ImageInfo;
        public readonly byte[] Buffer;
        public readonly IntPtr Pointer;
        public readonly SKBitmap TargetBitmap;
        public readonly SKCanvas Canvas;

        private readonly GCHandle _handle;

        public PinnedMemoryBuffer(SKImageInfo imageInfo)
        {
            ImageInfo = imageInfo;

            //var _imageInfo = new SKImageInfo(width, height, SKColorType.Rgb888x, SKAlphaType.Opaque);
            Buffer = new byte[imageInfo.BytesSize];

            _handle = GCHandle.Alloc(Buffer, GCHandleType.Pinned);
            Pointer = _handle.AddrOfPinnedObject();

            // Wrap the pinned buffer in a SKBitmap so we can draw into it
            TargetBitmap = new SKBitmap();
            if (!TargetBitmap.InstallPixels(imageInfo, Pointer, imageInfo.RowBytes))
                throw new Exception("Failed to install pixels into SKBitmap");

            Canvas = new SKCanvas(TargetBitmap);
        }

        public void Dispose()
        {
            Canvas?.Dispose();
            TargetBitmap?.Dispose();

            if (_handle.IsAllocated)
                _handle.Free();

            GC.SuppressFinalize(this);
        }
    }
}
