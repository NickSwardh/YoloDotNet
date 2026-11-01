// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Extensions
{
    public static class ImageResizeExtension
    {
        /// <summary>
        /// Resizes the input image to the target dimensions by stretching it to fit the model input size, returning a pointer to RGB888x pixel data and the new dimensions.
        /// </summary>
        /// <remarks>
        /// This method is intended for models trained on stretched (non-aspect-ratio-preserving) datasets.
        /// Using this with models trained on letterbox/proportional preprocessing may reduce inference accuracy.
        /// For standard models, use <see cref="ResizeImageProportional{T}"/> instead.
        /// </remarks>
        /// <param name="img">The original image to resize.</param>
        /// <param name="samplingOptions">Sampling options used during resizing.</param>
        /// <param name="pinnedMemoryBuffer">A pinned memory buffer where the resized image will be written.</param>
        /// <param name="roi">Optional region of interest to crop before resizing.</param>
        /// <returns>The dimensions of the input image (or ROI if specified), required for bounding box scaling.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static SKSizeI ResizeImageStretched<T>(this T img, SKSamplingOptions samplingOptions, PinnedMemoryBuffer pinnedMemoryBuffer, SKRectI? roi = null)
        {
            // Fast path for opaque 32-bit images with the default bilinear filter; everything else is drawn by SkiaSharp.
            if (TryResizeBilinear(img, samplingOptions, pinnedMemoryBuffer, roi, stretch: true, out var resizedSize))
                return resizedSize;

            var image = SourceImage(img, roi, out var createdImage);

            int modelWidth = pinnedMemoryBuffer.ImageInfo.Width;
            int modelHeight = pinnedMemoryBuffer.ImageInfo.Height;
            int width = image.Width;
            int height = image.Height;

            // Stretch the image to fit the model input size regardless of aspect ratio and cropped ROI.
            // This may distort the image but ensures it matches the model's expected input dimensions.
            var srcRect = new SKRect(0, 0, image.Width, image.Height);
            var destRect = new SKRect(0, 0, modelWidth, modelHeight);

            pinnedMemoryBuffer.Canvas.DrawImage(image, srcRect, destRect, samplingOptions);

            // Only dispose if a new SKImage was created for drawing, not the caller's own image
            if (createdImage)
                image.Dispose();

            // Return the input image dimensions (ROI dimensions if cropped), required to correctly scale bounding boxes
            return new SKSizeI(width, height);
        }

        /// <summary>
        /// Resizes the input image proportionally to fit the model input size, with RGB888x format and padded borders, returning a pointer to the pixel data and the new image dimensions.
        /// </summary>
        /// <param name="img">The original image to resize.</param>
        /// <param name="samplingOptions">Sampling options used during resizing.</param>
        /// <param name="pinnedMemoryBuffer">A pinned memory buffer where the resized image will be written.</param>
        /// <param name="roi">Optional region of interest to crop before resizing.</param>
        /// <returns>The dimensions of the input image (or ROI if specified), required for bounding box scaling.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static SKSizeI ResizeImageProportional<T>(this T img, SKSamplingOptions samplingOptions, PinnedMemoryBuffer pinnedMemoryBuffer, SKRectI? roi = null)
        {
            // Fast path for opaque 32-bit images with the default bilinear filter; everything else is drawn by SkiaSharp.
            if (TryResizeBilinear(img, samplingOptions, pinnedMemoryBuffer, roi, stretch: false, out var resizedSize))
                return resizedSize;

            var image = SourceImage(img, roi, out var createdImage);

            int width = image.Width;
            int height = image.Height;

            var srcRect = new SKRect(0, 0, width, height);
            var dstRect = (SKRect)ProportionalDestination(width, height, pinnedMemoryBuffer.ImageInfo.Width, pinnedMemoryBuffer.ImageInfo.Height);

            // Draw the resized image onto the pinned memory buffer canvas as RGB888x, padded with the letterbox colour
            pinnedMemoryBuffer.Canvas.DrawImage(image, srcRect, dstRect, samplingOptions);

            // Only dispose if a new SKImage was created for drawing, not the caller's own image
            if (createdImage)
                image.Dispose();

            // Return the input image dimensions (ROI dimensions if cropped), required to correctly scale bounding boxes
            return new SKSizeI(width, height);
        }

        /// <summary>
        /// The area of the model input that an image of the given size fills when it is scaled proportionally and centred.
        /// </summary>
        /// <remarks>
        /// Images smaller than the model input are scaled up as well, so objects reach the scale the model was trained on.
        /// </remarks>
        internal static SKRectI ProportionalDestination(int width, int height, int modelWidth, int modelHeight)
        {
            double scaleFactor = Math.Min((double)modelWidth / width, (double)modelHeight / height);

            // Round half to even in double precision, as the Ultralytics letterbox does, so that the
            // model receives the same input (e.g. 853 * 0.5 = 426.5 is resized to 426 rows, not 427).
            int newWidth = (int)Math.Round(width * scaleFactor, MidpointRounding.ToEven);
            int newHeight = (int)Math.Round(height * scaleFactor, MidpointRounding.ToEven);

            // Centre the image within the model dimensions
            int x = (modelWidth - newWidth) / 2;
            int y = (modelHeight - newHeight) / 2;

            return new SKRectI(x, y, x + newWidth, y + newHeight);
        }

        /// <summary>
        /// Labels 32-bit RGBA/BGRA pixels as opaque, so that their alpha channel is ignored and the stored colours are used.
        /// Other pixel formats are returned unchanged.
        /// </summary>
        /// <remarks>
        /// YOLO models are trained on images loaded without their alpha channel: a pixel's stored colour is used whatever
        /// its transparency, not blended with a background. Ignoring alpha here gives the model the same input.
        /// For images with transparency that requires the stored colours to still be there: decode them with
        /// <see cref="SKAlphaType.Unpremul"/>. SkiaSharp decodes them premultiplied by default, which has already
        /// multiplied each colour by its alpha (fully transparent pixels become black). Opaque images are unaffected.
        /// </remarks>
        private static SKImageInfo IgnoreAlpha(SKImageInfo info)
            => (info.ColorType == SKColorType.Bgra8888 || info.ColorType == SKColorType.Rgba8888) && info.AlphaType != SKAlphaType.Opaque
                ? info.WithAlphaType(SKAlphaType.Opaque)
                : info;

        /// <summary>
        /// The image SkiaSharp draws from: the input image (or its ROI), with the alpha channel of 32-bit pixels ignored.
        /// </summary>
        /// <param name="created">True when the returned image was created here and must be disposed by the caller.</param>
        private static SKImage SourceImage<T>(T img, SKRectI? roi, out bool created)
        {
            if (img is SKBitmap bitmap)
            {
                created = true;

                // ROIs that reach outside the bitmap are copied into a ROI-sized bitmap, as before
                if (roi.HasValue && !new SKRectI(0, 0, bitmap.Width, bitmap.Height).Contains(roi.Value))
                    return YoloCore.CropToRoi(bitmap, roi.Value);

                var image = SKImage.FromPixels(IgnoreAlpha(bitmap.Info), bitmap.GetPixels(), bitmap.RowBytes);

                if (!roi.HasValue)
                    return image;

                using (image)
                    return YoloCore.CropToRoi(image, roi.Value);
            }

            if (img is SKImage skImage)
            {
                // Images with raster pixels are relabelled; others (for example lazily decoded ones) are drawn as they are
                var source = skImage;
                created = false;

                using (var pixmap = skImage.PeekPixels())
                {
                    if (pixmap is not null && IgnoreAlpha(pixmap.Info).AlphaType != pixmap.AlphaType)
                    {
                        source = SKImage.FromPixels(IgnoreAlpha(pixmap.Info), pixmap.GetPixels(), pixmap.RowBytes);
                        created = true;
                    }
                }

                if (!roi.HasValue)
                    return source;

                var cropped = YoloCore.CropToRoi(source, roi.Value);

                if (created)
                    source.Dispose();

                created = true;
                return cropped;
            }

            throw new ArgumentException($"Unsupported image type {typeof(T).Name}. Use SKBitmap or SKImage.", nameof(img));
        }

        /// <summary>
        /// Resizes with <see cref="Avx2BilinearImageResizer"/> when the image and sampling options allow it.
        /// </summary>
        /// <returns>
        /// True when the image was resized, with the source size (the ROI size if specified) in <paramref name="size"/>.
        /// False when SkiaSharp should draw it instead; nothing has been written then.
        /// </returns>
        internal static bool TryResizeNormalized<T>(T img, SKSamplingOptions samplingOptions, float[] tensor,
            int width, int height, SKRectI? roi, bool stretch, out SKSizeI size)
            => TryResizeBilinear(img, samplingOptions, null, roi, stretch, out size, tensor, width, height);

        /// <summary>
        /// Half-precision (FP16) version of <see cref="TryResizeNormalized{T}(T, SKSamplingOptions, float[], int, int, SKRectI?, bool, out SKSizeI)"/>:
        /// gives the same values as resizing and then normalizing with the FP16 lookup table.
        /// </summary>
        internal static bool TryResizeNormalized<T>(T img, SKSamplingOptions samplingOptions, ushort[] tensor,
            int width, int height, SKRectI? roi, bool stretch, out SKSizeI size)
            => TryResizeBilinear(img, samplingOptions, null, roi, stretch, out size, null, width, height, tensor);

        private static bool TryResizeBilinear<T>(T img, SKSamplingOptions samplingOptions, PinnedMemoryBuffer? pinnedMemoryBuffer,
            SKRectI? roi, bool stretch, out SKSizeI size, float[]? tensor = null, int tensorWidth = 0, int tensorHeight = 0,
            ushort[]? halfTensor = null)
        {
            size = default;

            // Only the default filter, bilinear without mipmaps; any other sampling is left to SkiaSharp.
            if (!Avx2BilinearImageResizer.IsSupported
                || samplingOptions.UseCubic
                || samplingOptions.IsAniso
                || samplingOptions.Filter != SKFilterMode.Linear
                || samplingOptions.Mipmap != SKMipmapMode.None)
                return false;

            // Images without raster pixels (for example lazily decoded ones) have no pixmap and are drawn by SkiaSharp.
            using var pixmap = img switch
            {
                SKBitmap bitmap => bitmap.PeekPixels(),
                SKImage image => image.PeekPixels(),
                _ => null
            };

            if (pixmap is null)
                return false;

            // The same pixels, with the alpha channel ignored (see IgnoreAlpha)
            using var opaque = new SKPixmap(IgnoreAlpha(pixmap.Info), pixmap.GetPixels(), pixmap.RowBytes);

            // A ROI is read in place instead of being copied. ROIs that reach outside the image keep the SkiaSharp path.
            SKPixmap? subset = null;

            if (roi.HasValue)
            {
                if (!new SKRectI(0, 0, opaque.Width, opaque.Height).Contains(roi.Value))
                    return false;

                subset = opaque.ExtractSubset(roi.Value);

                if (subset is null)
                    return false;
            }

            using (subset)
            {
                var source = subset ?? opaque;
                int modelWidth = pinnedMemoryBuffer?.ImageInfo.Width ?? tensorWidth;
                int modelHeight = pinnedMemoryBuffer?.ImageInfo.Height ?? tensorHeight;

                var destination = stretch
                    ? new SKRectI(0, 0, modelWidth, modelHeight)
                    : ProportionalDestination(source.Width, source.Height, modelWidth, modelHeight);

                bool resized = tensor is not null
                    ? Avx2BilinearImageResizer.TryResizeNormalized(source, tensor, modelWidth, modelHeight, destination)
                    : halfTensor is not null
                        ? Avx2BilinearImageResizer.TryResizeNormalizedHalf(source, halfTensor, modelWidth, modelHeight, destination, _byteToNormalizedHalf)
                        : Avx2BilinearImageResizer.TryResize(source, pinnedMemoryBuffer!.Pointer, pinnedMemoryBuffer.ImageInfo.RowBytes, destination);

                if (!resized)
                    return false;

                // The input image dimensions (ROI dimensions if cropped), required to correctly scale bounding boxes
                size = new SKSizeI(source.Width, source.Height);
                return true;
            }
        }

        /// <summary>
        /// Converts raw pixel image data to a normalized float array for model input.
        /// </summary>
        /// <param name="pixelsPtr">A pointer to the raw pixel image data in memory.</param>
        /// <param name="inputShape">The shape of the input tensor.</param>
        /// <param name="tensorBufferSize">The size of the tensor buffer, which should be equal to the product of the input shape dimensions.</param>
        /// <param name="tensorArrayBuffer">A pre-allocated float array buffer to store the normalized pixel values.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        unsafe public static void NormalizePixelsToArray(this IntPtr pixelsPtr,
            long[] inputShape,
            int tensorBufferSize,
            float[] tensorArrayBuffer)
        {
            var colorChannels = (int)inputShape[1];
            var height = (int)inputShape[2];
            var width = (int)inputShape[3];
            int totalPixels = width * height;

            const float inv255 = 1.0f / 255.0f;
            byte* src = (byte*)pixelsPtr;

            // Pin the destination buffer for the duration of the write to avoid the GC
            // relocating the rented array while raw pointers are in use.
            fixed (float* dst = tensorArrayBuffer)
            {
                if (colorChannels == 1)
                    NormalizeGrayscaleToFloat(src, dst, totalPixels, inv255);
                else
                    NormalizeRgbToFloat(src, dst, totalPixels, inv255);
            }
        }

        /// <summary>
        /// Normalizes a single grayscale channel (read from the R component of the RGBX source) into a planar float buffer.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void NormalizeGrayscaleToFloat(byte* src, float* dst, int totalPixels, float inv255)
        {
            int srcIndex = 0;
            for (int i = 0; i < totalPixels; i++, srcIndex += 4)
                dst[i] = src[srcIndex] * inv255; // Read only the grayscale component (assumed in R channel)
        }

        /// <summary>
        /// Normalizes interleaved RGBX pixel data into three planar float channels (R, G, B).
        /// Uses SSSE3/SSE4.1 to de-interleave and convert four pixels per iteration when available,
        /// falling back to a scalar loop otherwise (and for the trailing pixels).
        /// </summary>
        private static unsafe void NormalizeRgbToFloat(byte* src, float* dst, int totalPixels, float inv255)
        {
            float* dstR = dst;
            float* dstG = dstR + totalPixels;
            float* dstB = dstG + totalPixels;

            int i = 0;

            if (Ssse3.IsSupported && Sse41.IsSupported)
            {
                var scale = Vector128.Create(inv255);

                // pshufb masks that gather one channel's byte from each of four RGBX pixels into the
                // low four bytes of the vector (0x80 selects a zero byte for the unused upper lanes).
                var rMask = Vector128.Create((byte)0, 4, 8, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
                var gMask = Vector128.Create((byte)1, 5, 9, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
                var bMask = Vector128.Create((byte)2, 6, 10, 14, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);

                int limit = totalPixels - 4;
                for (; i <= limit; i += 4)
                {
                    var pixels = Sse2.LoadVector128(src + i * 4); // four RGBX pixels (16 bytes)

                    var r = Sse2.ConvertToVector128Single(Sse41.ConvertToVector128Int32(Ssse3.Shuffle(pixels, rMask)));
                    var g = Sse2.ConvertToVector128Single(Sse41.ConvertToVector128Int32(Ssse3.Shuffle(pixels, gMask)));
                    var b = Sse2.ConvertToVector128Single(Sse41.ConvertToVector128Int32(Ssse3.Shuffle(pixels, bMask)));

                    Sse.Store(dstR + i, Sse.Multiply(r, scale));
                    Sse.Store(dstG + i, Sse.Multiply(g, scale));
                    Sse.Store(dstB + i, Sse.Multiply(b, scale));
                }
            }

            // Scalar tail (and full fallback when SIMD is unavailable).
            for (; i < totalPixels; i++)
            {
                int srcIndex = i * 4;
                dstR[i] = src[srcIndex] * inv255;
                dstG[i] = src[srcIndex + 1] * inv255;
                dstB[i] = src[srcIndex + 2] * inv255;
            }
        }

        /// <summary>
        /// Overload of NormalizePixelsToArray that converts raw pixel image data to a normalized half-precision float (ushort) array for model input.
        /// </summary>
        /// <param name="pixelsPtr"></param>
        /// <param name="inputShape"></param>
        /// <param name="tensorBufferSize"></param>
        /// <param name="tensorArrayBuffer"></param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        unsafe public static void NormalizePixelsToArray(this IntPtr pixelsPtr,
            long[] inputShape,
            int tensorBufferSize,
            ushort[] tensorArrayBuffer)
        {
            var colorChannels = (int)inputShape[1];
            var height = (int)inputShape[2];
            var width = (int)inputShape[3];
            int totalPixels = width * height;

            byte* src = (byte*)pixelsPtr;

            // Pin the destination buffer and the lookup table for the duration of the write to avoid the GC
            // relocating them while raw pointers are in use.
            fixed (ushort* dst = tensorArrayBuffer)
            fixed (ushort* byteToHalf = _byteToNormalizedHalf)
            {
                if (colorChannels == 1)
                {
                    int srcIndex = 0;
                    for (int i = 0; i < totalPixels; i++, srcIndex += 4)
                        dst[i] = byteToHalf[src[srcIndex]];
                }
                else
                {
                    ushort* dstR = dst;
                    ushort* dstG = dstR + totalPixels;
                    ushort* dstB = dstG + totalPixels;

                    int srcIndex = 0;
                    for (int i = 0; i < totalPixels; i++, srcIndex += 4)
                    {
                        dstR[i] = byteToHalf[src[srcIndex]];
                        dstG[i] = byteToHalf[src[srcIndex + 1]];
                        dstB[i] = byteToHalf[src[srcIndex + 2]];
                    }
                }
            }
        }

        // Normalized half-precision value for every possible byte: FloatToUshort(b / 255) for b = 0..255.
        // A pixel channel is a byte, so a table lookup replaces the per-value float-to-half conversion.
        private static readonly ushort[] _byteToNormalizedHalf = BuildByteToNormalizedHalfTable();

        private static ushort[] BuildByteToNormalizedHalfTable()
        {
            const float inv255 = 1.0f / 255.0f;
            var table = new ushort[256];

            for (int b = 0; b < table.Length; b++)
                table[b] = FloatToUshort(b * inv255);

            return table;
        }

        // Helper method to convert float to half-precision (16-bit) float (ushort)
        unsafe private static ushort FloatToUshort(float value)
        {
            // Avoid BitConverter for performance reasons and use unsafe cast instead.
            uint f = *(uint*)&value;

            // Extract parts
            int sign = (int)(f >> 16) & 0x8000;
            int exponent = (int)((f >> 23) & 0xFF) - 112;
            int mantissa = (int)(f & 0x7FFFFF);

            if (exponent <= 0)
            {
                if (exponent < -10)
                {
                    return (ushort)sign; // too small -> zero
                }
                mantissa = (mantissa | 0x800000) >> (1 - exponent);
                return (ushort)(sign | (mantissa + 0xFFF + ((mantissa >> 13) & 1)) >> 13);
            }
            else if (exponent == 143 - 112) // Inf/NaN
            {
                if (mantissa == 0)
                    return (ushort)(sign | 0x7C00); // Inf
                return (ushort)(sign | 0x7C00 | (mantissa >> 13)); // NaN
            }
            else
            {
                if (exponent > 30)
                {
                    return (ushort)(sign | 0x7C00); // overflow -> Inf
                }
                return (ushort)(sign | (exponent << 10) | (mantissa + 0xFFF + ((mantissa >> 13) & 1)) >> 13);
            }
        }
    }
}
