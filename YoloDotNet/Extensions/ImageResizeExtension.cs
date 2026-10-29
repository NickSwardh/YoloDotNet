// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Extensions
{
    public static class ImageResizeExtension
    {
        private static readonly SKSamplingOptions LinearNoMipmap = new(SKFilterMode.Linear, SKMipmapMode.None);

        private static readonly Vector128<byte> ShuffleR = Vector128.Create((byte)0, 4, 8, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
        private static readonly Vector128<byte> ShuffleG = Vector128.Create((byte)1, 5, 9, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
        private static readonly Vector128<byte> ShuffleB = Vector128.Create((byte)2, 6, 10, 14, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
        private static readonly Vector128<byte> ShuffleRFromBgra = Vector128.Create((byte)2, 6, 10, 14, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
        private static readonly Vector128<byte> ShuffleBFromBgra = Vector128.Create((byte)0, 4, 8, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);

        /// <summary>
        /// Resizes the input image to the target dimensions by stretching it to fit the model input size, returning a pointer to RGB888x pixel data and the new dimensions.
        /// </summary>
        /// <remarks>
        /// This method is intended for models trained on stretched (non-aspect-ratio-preserving) datasets.
        /// Using this with models trained on letterbox/proportional preprocessing may reduce inference accuracy.
        /// For standard models, use <see cref="ResizeImageProportional{T}"/> instead.
        /// Destination size comes from <paramref name="pinnedMemoryBuffer"/> (ONNX input H×W), not a hardcoded value.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static SKSizeI ResizeImageStretched<T>(this T img, SKSamplingOptions samplingOptions, PinnedMemoryBuffer pinnedMemoryBuffer, SKRectI? roi = null)
        {
            int modelWidth = pinnedMemoryBuffer.ImageInfo.Width;
            int modelHeight = pinnedMemoryBuffer.ImageInfo.Height;

            // ROI 直接从原图采样到模型输入。先裁出整块再缩放会多两次全分辨率拷贝。
            if (img is SKBitmap bitmap
                && TryDrawBitmapStretched(bitmap, roi, pinnedMemoryBuffer.Canvas, samplingOptions, modelWidth, modelHeight, out var sourceSize))
                return sourceSize;

            var (image, createdImage) = CreateSourceImage(img, roi);

            int width = image.Width;
            int height = image.Height;

            DrawResized(image, pinnedMemoryBuffer.Canvas, samplingOptions, modelWidth, modelHeight, destX: 0, destY: 0);

            if (createdImage || roi.HasValue)
                image.Dispose();

            return new SKSizeI(width, height);
        }

        static bool TryDrawBitmapStretched(
            SKBitmap bitmap,
            SKRectI? roi,
            SKCanvas canvas,
            SKSamplingOptions samplingOptions,
            int modelWidth,
            int modelHeight,
            out SKSizeI sourceSize)
        {
            var source = roi ?? new SKRectI(0, 0, bitmap.Width, bitmap.Height);
            sourceSize = new SKSizeI(source.Width, source.Height);
            if (source.Width < 1 || source.Height < 1 || bitmap.GetPixels() == IntPtr.Zero)
                return false;
            if (NeedsPyramid(samplingOptions, source.Width, source.Height, modelWidth, modelHeight))
                return false;

            using var image = SKImage.FromPixels(bitmap.Info, bitmap.GetPixels());
            canvas.DrawImage(
                image,
                new SKRect(source.Left, source.Top, source.Right, source.Bottom),
                new SKRect(0, 0, modelWidth, modelHeight),
                samplingOptions);
            return true;
        }

        /// <summary>
        /// Resizes the input image proportionally to fit the model input size, with RGB888x format and padded borders, returning a pointer to the pixel data and the new image dimensions.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static SKSizeI ResizeImageProportional<T>(this T img, SKSamplingOptions samplingOptions, PinnedMemoryBuffer pinnedMemoryBuffer, SKRectI? roi = null)
        {
            var (image, createdImage) = CreateSourceImage(img, roi);

            int modelWidth = pinnedMemoryBuffer.ImageInfo.Width;
            int modelHeight = pinnedMemoryBuffer.ImageInfo.Height;
            int width = image.Width;
            int height = image.Height;

            if (width < modelWidth && height < modelHeight)
            {
                int x = (modelWidth - width) / 2;
                int y = (modelHeight - height) / 2;
                pinnedMemoryBuffer.Canvas.DrawImage(
                    image,
                    new SKRect(0, 0, width, height),
                    new SKRect(x, y, x + width, y + height),
                    samplingOptions);
            }
            else
            {
                float scaleFactor = Math.Min((float)modelWidth / width, (float)modelHeight / height);
                int newWidth = (int)((width * scaleFactor) + 0.5f);
                int newHeight = (int)((height * scaleFactor) + 0.5f);
                int x = (modelWidth - newWidth) / 2;
                int y = (modelHeight - newHeight) / 2;

                DrawResized(image, pinnedMemoryBuffer.Canvas, samplingOptions, newWidth, newHeight, x, y);
            }

            if (createdImage || roi.HasValue)
                image.Dispose();

            return new SKSizeI(width, height);
        }

        /// <summary>
        /// Copies the image into the model input buffer without scaling.
        /// The source must already match the model input size.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static SKSizeI CopyImage<T>(this T img, PinnedMemoryBuffer pinnedMemoryBuffer, SKRectI? roi = null)
        {
            var (image, createdImage) = CreateSourceImage(img, roi);

            int modelWidth = pinnedMemoryBuffer.ImageInfo.Width;
            int modelHeight = pinnedMemoryBuffer.ImageInfo.Height;
            int width = image.Width;
            int height = image.Height;

            try
            {
                if (width != modelWidth || height != modelHeight)
                    throw new YoloDotNetException($"Image size {width}x{height} does not match model input {modelWidth}x{modelHeight}.");

                pinnedMemoryBuffer.Canvas.DrawImage(image, 0, 0);
                return new SKSizeI(width, height);
            }
            finally
            {
                if (createdImage || roi.HasValue)
                    image.Dispose();
            }
        }

        /// <summary>
        /// 源图（或 ROI）已经是模型输入尺寸时，直接把像素归一化写入 NCHW，跳过 Canvas 中转。
        /// 只接受不透明的 Rgba8888 / Rgb888x / Bgra8888 / Gray8。其余格式返回 false，由调用方走绘制路径。
        /// </summary>
        public static bool TryNormalizeMatched<T>(
            this T image,
            int modelWidth,
            int modelHeight,
            int channels,
            float[] destination,
            out SKSizeI sourceSize,
            SKRectI? roi = null,
            float[]? mean = null,
            float[]? std = null)
            => TryNormalizeMatched(image, modelWidth, modelHeight, channels, roi, mean, std, destination, null, out sourceSize);

        /// <summary>
        /// <see cref="TryNormalizeMatched{T}(T, int, int, int, float[], out SKSizeI, SKRectI?, float[], float[])"/> 的半精度版本。
        /// </summary>
        public static bool TryNormalizeMatched<T>(
            this T image,
            int modelWidth,
            int modelHeight,
            int channels,
            ushort[] destination,
            out SKSizeI sourceSize,
            SKRectI? roi = null,
            float[]? mean = null,
            float[]? std = null)
            => TryNormalizeMatched(image, modelWidth, modelHeight, channels, roi, mean, std, null, destination, out sourceSize);

        private static bool TryNormalizeMatched<T>(
            T image,
            int modelWidth,
            int modelHeight,
            int channels,
            SKRectI? roi,
            float[]? mean,
            float[]? std,
            float[]? floatDestination,
            ushort[]? halfDestination,
            out SKSizeI sourceSize)
        {
            sourceSize = default;
            if (modelWidth <= 0 || modelHeight <= 0 || channels is not (1 or 3))
                return false;

            var needed = modelWidth * modelHeight * channels;
            if (floatDestination is not null && floatDestination.Length < needed)
                return false;
            if (halfDestination is not null && halfDestination.Length < needed)
                return false;
            if (floatDestination is null && halfDestination is null)
                return false;

            var pixels = GetRasterPixels(image, out var info, out var rowBytes, out var pixmap);
            try
            {
                if (pixels == IntPtr.Zero || rowBytes <= 0)
                    return false;

                var colorSpace = info.ColorSpace;
                if (colorSpace is not null && !colorSpace.IsSrgb)
                    return false;

                if (info.AlphaType is not (SKAlphaType.Opaque or SKAlphaType.Unpremul))
                    return false;

                var originX = 0;
                var originY = 0;
                var width = info.Width;
                var height = info.Height;
                if (roi.HasValue)
                {
                    var region = roi.Value;
                    originX = region.Left;
                    originY = region.Top;
                    width = region.Width;
                    height = region.Height;
                }

                if (width != modelWidth || height != modelHeight)
                    return false;
                if (originX < 0 || originY < 0 || width <= 0 || height <= 0)
                    return false;
                if (originX > info.Width - width || originY > info.Height - height)
                    return false;

                var bytesPerPixel = info.BytesPerPixel;
                if ((long)originX * bytesPerPixel + (long)width * bytesPerPixel > rowBytes)
                    return false;

                ResolveScaleBias(mean, std, out var scaleR, out var scaleG, out var scaleB, out var biasR, out var biasG, out var biasB);
                unsafe
                {
                    var src = (byte*)pixels + (long)originY * rowBytes + (long)originX * bytesPerPixel;
                    if (channels == 1)
                    {
                        if (info.ColorType != SKColorType.Gray8 || bytesPerPixel != 1)
                            return false;

                        if (halfDestination is not null)
                        {
                            fixed (ushort* dst = halfDestination)
                                NormalizeGrayHalfRows(src, rowBytes, width, height, dst, scaleR, biasR);
                        }
                        else
                        {
                            fixed (float* dst = floatDestination)
                                NormalizeGrayRows(src, rowBytes, width, height, dst, scaleR, biasR);
                        }
                    }
                    else
                    {
                        int red;
                        int blue;
                        Vector128<byte> shuffleR;
                        Vector128<byte> shuffleB;
                        switch (info.ColorType)
                        {
                            case SKColorType.Rgba8888:
                            case SKColorType.Rgb888x:
                                red = 0;
                                blue = 2;
                                shuffleR = ShuffleR;
                                shuffleB = ShuffleB;
                                break;
                            case SKColorType.Bgra8888:
                                red = 2;
                                blue = 0;
                                shuffleR = ShuffleRFromBgra;
                                shuffleB = ShuffleBFromBgra;
                                break;
                            default:
                                return false;
                        }

                        if (bytesPerPixel != 4)
                            return false;

                        if (halfDestination is not null)
                        {
                            fixed (ushort* dst = halfDestination)
                                NormalizeRgbHalfRows(src, rowBytes, width, height, dst, red, 1, blue, scaleR, scaleG, scaleB, biasR, biasG, biasB);
                        }
                        else
                        {
                            fixed (float* dst = floatDestination)
                                NormalizeRgbRows(src, rowBytes, width, height, dst, shuffleR, ShuffleG, shuffleB, red, 1, blue, scaleR, scaleG, scaleB, biasR, biasG, biasB);
                        }
                    }
                }

                sourceSize = new SKSizeI(width, height);
                return true;
            }
            finally
            {
                pixmap?.Dispose();
            }
        }

        private static IntPtr GetRasterPixels<T>(T image, out SKImageInfo info, out int rowBytes, out SKPixmap? pixmap)
        {
            switch (image)
            {
                case SKBitmap bitmap:
                    pixmap = null;
                    info = bitmap.Info;
                    rowBytes = bitmap.RowBytes;
                    return bitmap.GetPixels();
                case SKImage skImage:
                    pixmap = skImage.PeekPixels();
                    if (pixmap is null)
                    {
                        info = default;
                        rowBytes = 0;
                        return IntPtr.Zero;
                    }

                    info = pixmap.Info;
                    rowBytes = pixmap.RowBytes;
                    return pixmap.GetPixels();
                default:
                    pixmap = null;
                    info = default;
                    rowBytes = 0;
                    return IntPtr.Zero;
            }
        }

        private static unsafe void NormalizeRgbRows(
            byte* src,
            int stride,
            int width,
            int height,
            float* dst,
            Vector128<byte> shuffleR,
            Vector128<byte> shuffleG,
            Vector128<byte> shuffleB,
            int redOffset,
            int greenOffset,
            int blueOffset,
            float scaleR,
            float scaleG,
            float scaleB,
            float biasR,
            float biasG,
            float biasB)
        {
            var plane = width * height;
            float* dstR = dst;
            float* dstG = dst + plane;
            float* dstB = dstG + plane;
            var vector = Avx.IsSupported && Ssse3.IsSupported && Sse41.IsSupported;
            var vScaleR = vector ? Vector256.Create(scaleR) : default;
            var vScaleG = vector ? Vector256.Create(scaleG) : default;
            var vScaleB = vector ? Vector256.Create(scaleB) : default;
            var vBiasR = vector ? Vector256.Create(biasR) : default;
            var vBiasG = vector ? Vector256.Create(biasG) : default;
            var vBiasB = vector ? Vector256.Create(biasB) : default;
            var pixel = 0;

            for (var y = 0; y < height; y++)
            {
                var row = src + y * stride;
                var x = 0;

                if (vector)
                {
                    for (; x <= width - 8; x += 8, pixel += 8)
                    {
                        var p = row + (x << 2);
                        var px0 = Sse2.LoadVector128(p);
                        var px1 = Sse2.LoadVector128(p + 16);
                        var r = Vector256.Create(BytesToFloats(px0, shuffleR), BytesToFloats(px1, shuffleR));
                        var g = Vector256.Create(BytesToFloats(px0, shuffleG), BytesToFloats(px1, shuffleG));
                        var b = Vector256.Create(BytesToFloats(px0, shuffleB), BytesToFloats(px1, shuffleB));
                        Avx.Store(dstR + pixel, MultiplyAdd(r, vScaleR, vBiasR));
                        Avx.Store(dstG + pixel, MultiplyAdd(g, vScaleG, vBiasG));
                        Avx.Store(dstB + pixel, MultiplyAdd(b, vScaleB, vBiasB));
                    }
                }

                for (; x < width; x++, pixel++)
                {
                    var p = row + (x << 2);
                    dstR[pixel] = p[redOffset] * scaleR + biasR;
                    dstG[pixel] = p[greenOffset] * scaleG + biasG;
                    dstB[pixel] = p[blueOffset] * scaleB + biasB;
                }
            }
        }

        private static unsafe void NormalizeGrayRows(byte* src, int stride, int width, int height, float* dst, float scale, float bias)
        {
            var pixel = 0;
            for (var y = 0; y < height; y++)
            {
                var row = src + y * stride;
                for (var x = 0; x < width; x++, pixel++)
                    dst[pixel] = row[x] * scale + bias;
            }
        }

        private static unsafe void NormalizeRgbHalfRows(
            byte* src,
            int stride,
            int width,
            int height,
            ushort* dst,
            int redOffset,
            int greenOffset,
            int blueOffset,
            float scaleR,
            float scaleG,
            float scaleB,
            float biasR,
            float biasG,
            float biasB)
        {
            var plane = width * height;
            ushort* dstR = dst;
            ushort* dstG = dst + plane;
            ushort* dstB = dstG + plane;
            var pixel = 0;

            for (var y = 0; y < height; y++)
            {
                var row = src + y * stride;
                for (var x = 0; x < width; x++, pixel++)
                {
                    var p = row + (x << 2);
                    dstR[pixel] = FloatToUshort(p[redOffset] * scaleR + biasR);
                    dstG[pixel] = FloatToUshort(p[greenOffset] * scaleG + biasG);
                    dstB[pixel] = FloatToUshort(p[blueOffset] * scaleB + biasB);
                }
            }
        }

        private static unsafe void NormalizeGrayHalfRows(byte* src, int stride, int width, int height, ushort* dst, float scale, float bias)
        {
            var pixel = 0;
            for (var y = 0; y < height; y++)
            {
                var row = src + y * stride;
                for (var x = 0; x < width; x++, pixel++)
                    dst[pixel] = FloatToUshort(row[x] * scale + bias);
            }
        }

        /// <summary>
        /// Converts raw pixel image data to a normalized float array for model input.
        /// Channel count and spatial size are taken from <paramref name="inputShape"/> (NCHW).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        unsafe public static void NormalizePixelsToArray(this IntPtr pixelsPtr,
            long[] inputShape,
            int tensorBufferSize,
            float[] tensorArrayBuffer,
            float[]? mean = null,
            float[]? std = null)
        {
            var colorChannels = (int)inputShape[1];
            var height = (int)inputShape[2];
            var width = (int)inputShape[3];
            int totalPixels = width * height;
            byte* src = (byte*)pixelsPtr;

            ResolveScaleBias(mean, std, out var scaleR, out var scaleG, out var scaleB, out var biasR, out var biasG, out var biasB);

            fixed (float* dst = tensorArrayBuffer)
            {
                if (colorChannels == 1)
                    NormalizeGray(src, dst, totalPixels, scaleR, biasR);
                else
                    NormalizeRgb(src, dst, totalPixels, scaleR, scaleG, scaleB, biasR, biasG, biasB);
            }
        }

        /// <summary>
        /// Overload of NormalizePixelsToArray that converts raw pixel image data to a normalized half-precision float (ushort) array for model input.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        unsafe public static void NormalizePixelsToArray(this IntPtr pixelsPtr,
            long[] inputShape,
            int tensorBufferSize,
            ushort[] tensorArrayBuffer,
            float[]? mean = null,
            float[]? std = null)
        {
            var colorChannels = (int)inputShape[1];
            var height = (int)inputShape[2];
            var width = (int)inputShape[3];
            int totalPixels = width * height;
            byte* src = (byte*)pixelsPtr;

            ResolveScaleBias(mean, std, out var scaleR, out var scaleG, out var scaleB, out var biasR, out var biasG, out var biasB);

            fixed (ushort* dst = tensorArrayBuffer)
            {
                if (colorChannels == 1)
                    NormalizeGrayHalf(src, dst, totalPixels, scaleR, biasR);
                else
                    NormalizeRgbHalf(src, dst, totalPixels, scaleR, scaleG, scaleB, biasR, biasG, biasB);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static (SKImage image, bool createdImage) CreateSourceImage<T>(T img, SKRectI? roi)
        {
            if (img is SKImage skImage)
            {
                return roi.HasValue
                    ? (YoloCore.CropToRoi(skImage, (SKRectI)roi), true)
                    : (skImage, false);
            }

            if (img is SKBitmap skBitmap)
            {
                var image = roi.HasValue
                    ? YoloCore.CropToRoi(skBitmap, (SKRectI)roi)
                    : SKImage.FromPixels(skBitmap.Info, skBitmap.GetPixels());
                return (image, true);
            }

            throw new YoloDotNetException($"Unsupported image type: {typeof(T).Name}");
        }

        /// <summary>
        /// Draws <paramref name="image"/> into a destination rectangle of the model's actual input size.
        /// When mipmaps are requested and the source is more than 2× the destination, build an explicit
        /// box-filter pyramid relative to that destination (not a fixed size) so quality matches
        /// Linear+Mipmap without generating mipmaps for the full-resolution original.
        /// </summary>
        private static void DrawResized(
            SKImage image,
            SKCanvas canvas,
            SKSamplingOptions samplingOptions,
            int destWidth,
            int destHeight,
            int destX,
            int destY)
        {
            var destRect = new SKRect(destX, destY, destX + destWidth, destY + destHeight);

            if (!NeedsPyramid(samplingOptions, image.Width, image.Height, destWidth, destHeight))
            {
                canvas.DrawImage(image, new SKRect(0, 0, image.Width, image.Height), destRect, samplingOptions);
                return;
            }

            using var pyramid = BuildPyramid(image, destWidth, destHeight);
            using var drawImage = SKImage.FromPixels(pyramid.Info, pyramid.GetPixels());
            canvas.DrawImage(
                drawImage,
                new SKRect(0, 0, pyramid.Width, pyramid.Height),
                destRect,
                new SKSamplingOptions(samplingOptions.Filter, SKMipmapMode.None));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool NeedsPyramid(SKSamplingOptions samplingOptions, int srcW, int srcH, int destW, int destH)
            => samplingOptions.Mipmap != SKMipmapMode.None
               && destW > 0
               && destH > 0
               && (srcW > destW * 2 || srcH > destH * 2);

        private static SKBitmap BuildPyramid(SKImage source, int destW, int destH)
        {
            using var pixmap = source.PeekPixels();
            SKBitmap current = pixmap is not null
                ? DownsampleStep(pixmap, destW, destH)
                : DownsampleStep(SKBitmap.FromImage(source), destW, destH, disposeSource: true);

            while (current.Width > destW * 2 || current.Height > destH * 2)
            {
                var next = DownsampleStep(current, destW, destH, disposeSource: false);
                current.Dispose();
                current = next;
            }

            return current;
        }

        private static SKBitmap DownsampleStep(SKBitmap source, int destW, int destH, bool disposeSource)
        {
            try
            {
                using var pixmap = source.PeekPixels();
                if (pixmap is null)
                    throw new YoloDotNetException("Failed to read source pixels for pyramid downsample.");

                return DownsampleStep(pixmap, destW, destH);
            }
            finally
            {
                if (disposeSource)
                    source.Dispose();
            }
        }

        private static SKBitmap DownsampleStep(SKPixmap source, int destW, int destH)
        {
            int nextW = source.Width > destW * 2 ? Math.Max(destW, source.Width / 2) : source.Width;
            int nextH = source.Height > destH * 2 ? Math.Max(destH, source.Height / 2) : source.Height;

            if (nextW == source.Width && nextH == source.Height)
            {
                var copy = new SKBitmap(source.Info);
                using var destPixmap = copy.PeekPixels();
                if (destPixmap is not null)
                    source.ReadPixels(destPixmap);
                return copy;
            }

            var dst = new SKBitmap(source.Info.WithSize(nextW, nextH));
            bool exactHalf = nextW * 2 == source.Width
                             && nextH * 2 == source.Height
                             && source.Info.BytesPerPixel is 1 or 4;

            if (exactHalf)
            {
                BoxAverage2x2(
                    source.GetPixels(),
                    source.RowBytes,
                    dst.GetPixels(),
                    dst.RowBytes,
                    nextW,
                    nextH,
                    source.Info.BytesPerPixel);
            }
            else
            {
                using var destPixmap = dst.PeekPixels();
                if (destPixmap is null || !source.ScalePixels(destPixmap, LinearNoMipmap))
                    throw new YoloDotNetException("Failed to downsample image to model input size.");
            }

            return dst;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void BoxAverage2x2(
            IntPtr srcPtr,
            int srcStride,
            IntPtr dstPtr,
            int dstStride,
            int dstW,
            int dstH,
            int bytesPerPixel)
        {
            byte* src = (byte*)srcPtr;
            byte* dst = (byte*)dstPtr;

            if (bytesPerPixel == 4)
            {
                for (int y = 0; y < dstH; y++)
                {
                    byte* row0 = src + (y * 2) * srcStride;
                    byte* row1 = row0 + srcStride;
                    byte* dest = dst + y * dstStride;

                    for (int x = 0; x < dstW; x++)
                    {
                        int s = x * 8;
                        int d = x * 4;
                        dest[d] = (byte)((row0[s] + row0[s + 4] + row1[s] + row1[s + 4]) >> 2);
                        dest[d + 1] = (byte)((row0[s + 1] + row0[s + 5] + row1[s + 1] + row1[s + 5]) >> 2);
                        dest[d + 2] = (byte)((row0[s + 2] + row0[s + 6] + row1[s + 2] + row1[s + 6]) >> 2);
                        dest[d + 3] = (byte)((row0[s + 3] + row0[s + 7] + row1[s + 3] + row1[s + 7]) >> 2);
                    }
                }

                return;
            }

            for (int y = 0; y < dstH; y++)
            {
                byte* row0 = src + (y * 2) * srcStride;
                byte* row1 = row0 + srcStride;
                byte* dest = dst + y * dstStride;

                for (int x = 0; x < dstW; x++)
                    dest[x] = (byte)((row0[x * 2] + row0[x * 2 + 1] + row1[x * 2] + row1[x * 2 + 1]) >> 2);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ResolveScaleBias(
            float[]? mean,
            float[]? std,
            out float scaleR,
            out float scaleG,
            out float scaleB,
            out float biasR,
            out float biasG,
            out float biasB)
        {
            const float inv255 = 1f / 255f;

            if (mean is { Length: >= 3 } && std is { Length: >= 3 })
            {
                scaleR = inv255 / std[0];
                scaleG = inv255 / std[1];
                scaleB = inv255 / std[2];
                biasR = -mean[0] / std[0];
                biasG = -mean[1] / std[1];
                biasB = -mean[2] / std[2];
                return;
            }

            scaleR = scaleG = scaleB = inv255;
            biasR = biasG = biasB = 0f;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void NormalizeRgb(
            byte* src,
            float* dst,
            int totalPixels,
            float scaleR,
            float scaleG,
            float scaleB,
            float biasR,
            float biasG,
            float biasB)
        {
            float* dstR = dst;
            float* dstG = dst + totalPixels;
            float* dstB = dstG + totalPixels;
            int i = 0;

            if (Avx.IsSupported && Ssse3.IsSupported && Sse41.IsSupported)
            {
                var vScaleR = Vector256.Create(scaleR);
                var vScaleG = Vector256.Create(scaleG);
                var vScaleB = Vector256.Create(scaleB);
                var vBiasR = Vector256.Create(biasR);
                var vBiasG = Vector256.Create(biasG);
                var vBiasB = Vector256.Create(biasB);

                for (; i <= totalPixels - 8; i += 8, src += 32)
                {
                    var px0 = Sse2.LoadVector128(src);
                    var px1 = Sse2.LoadVector128(src + 16);

                    var r = Vector256.Create(BytesToFloats(px0, ShuffleR), BytesToFloats(px1, ShuffleR));
                    var g = Vector256.Create(BytesToFloats(px0, ShuffleG), BytesToFloats(px1, ShuffleG));
                    var b = Vector256.Create(BytesToFloats(px0, ShuffleB), BytesToFloats(px1, ShuffleB));

                    Avx.Store(dstR + i, MultiplyAdd(r, vScaleR, vBiasR));
                    Avx.Store(dstG + i, MultiplyAdd(g, vScaleG, vBiasG));
                    Avx.Store(dstB + i, MultiplyAdd(b, vScaleB, vBiasB));
                }
            }

            for (; i < totalPixels; i++, src += 4)
            {
                dstR[i] = src[0] * scaleR + biasR;
                dstG[i] = src[1] * scaleG + biasG;
                dstB[i] = src[2] * scaleB + biasB;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void NormalizeGray(byte* src, float* dst, int totalPixels, float scale, float bias)
        {
            int i = 0;

            if (Avx.IsSupported && Ssse3.IsSupported && Sse41.IsSupported)
            {
                var vScale = Vector256.Create(scale);
                var vBias = Vector256.Create(bias);

                for (; i <= totalPixels - 8; i += 8, src += 32)
                {
                    var px0 = Sse2.LoadVector128(src);
                    var px1 = Sse2.LoadVector128(src + 16);
                    var gray = Vector256.Create(BytesToFloats(px0, ShuffleR), BytesToFloats(px1, ShuffleR));
                    Avx.Store(dst + i, MultiplyAdd(gray, vScale, vBias));
                }
            }

            for (; i < totalPixels; i++, src += 4)
                dst[i] = src[0] * scale + bias;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void NormalizeRgbHalf(
            byte* src,
            ushort* dst,
            int totalPixels,
            float scaleR,
            float scaleG,
            float scaleB,
            float biasR,
            float biasG,
            float biasB)
        {
            ushort* dstR = dst;
            ushort* dstG = dst + totalPixels;
            ushort* dstB = dstG + totalPixels;

            for (int i = 0; i < totalPixels; i++, src += 4)
            {
                dstR[i] = FloatToUshort(src[0] * scaleR + biasR);
                dstG[i] = FloatToUshort(src[1] * scaleG + biasG);
                dstB[i] = FloatToUshort(src[2] * scaleB + biasB);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void NormalizeGrayHalf(byte* src, ushort* dst, int totalPixels, float scale, float bias)
        {
            for (int i = 0; i < totalPixels; i++, src += 4)
                dst[i] = FloatToUshort(src[0] * scale + bias);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector128<float> BytesToFloats(Vector128<byte> pixels, Vector128<byte> shuffle)
            => Sse2.ConvertToVector128Single(Sse41.ConvertToVector128Int32(Ssse3.Shuffle(pixels, shuffle)));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<float> MultiplyAdd(Vector256<float> value, Vector256<float> scale, Vector256<float> bias)
            => Fma.IsSupported
                ? Fma.MultiplyAdd(value, scale, bias)
                : Avx.Add(Avx.Multiply(value, scale), bias);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static ushort FloatToUshort(float value)
            => BitConverter.HalfToUInt16Bits((Half)value);
    }
}
