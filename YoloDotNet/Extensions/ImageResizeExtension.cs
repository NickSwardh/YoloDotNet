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
            SKImage image = default!;
            var createdImage = false;

            if (img is SKImage skImage)
            {
                image = roi.HasValue
                    ? YoloCore.CropToRoi(skImage, (SKRectI)roi)
                    : skImage;
            }
            else if (img is SKBitmap skBitmap)
            {
                image = roi.HasValue
                    ? YoloCore.CropToRoi(skBitmap, (SKRectI)roi)
                    : SKImage.FromPixels(skBitmap.Info, skBitmap.GetPixels());

                createdImage = true;
            }

            int modelWidth = pinnedMemoryBuffer.ImageInfo.Width;
            int modelHeight = pinnedMemoryBuffer.ImageInfo.Height;
            int width = image.Width;
            int height = image.Height;

            // Stretch the image to fit the model input size regardless of aspect ratio and cropped ROI.
            // This may distort the image but ensures it matches the model's expected input dimensions.
            var srcRect = new SKRect(0, 0, image.Width, image.Height);
            var destRect = new SKRect(0, 0, modelWidth, modelHeight);

            pinnedMemoryBuffer.Canvas.DrawImage(image, srcRect, destRect, samplingOptions);

            // Only dispose if we created a new SKImage from SKBitmap or if we cropped to a ROI, since cropping creates a new SKImage instance
            if (createdImage || roi.HasValue)
                image?.Dispose();

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
            SKImage image = default!;
            var createdImage = false;

            if (img is SKImage skImage)
            {
                image = roi.HasValue
                    ? YoloCore.CropToRoi(skImage, (SKRectI)roi)
                    : skImage;
            }
            else if (img is SKBitmap skBitmap)
            {
                image = roi.HasValue
                    ? YoloCore.CropToRoi(skBitmap, (SKRectI)roi)
                    : SKImage.FromPixels(skBitmap.Info, skBitmap.GetPixels());

                createdImage = true;
            }

            int modelWidth = pinnedMemoryBuffer.ImageInfo.Width;
            int modelHeight = pinnedMemoryBuffer.ImageInfo.Height;
            int width = image.Width;
            int height = image.Height;

            // If the image is smaller than the model input size, we can draw it directly onto the pinned memory buffer canvas without resizing, which avoids unnecessary resampling and preserves image quality.
            if (width < modelWidth && height < modelHeight)
            {
                int x = (modelWidth - width) / 2;
                int y = (modelHeight - height) / 2;
                var srcRect = new SKRect(0, 0, width, height);
                var dstRect = new SKRect(x, y, x + width, y + height);
                pinnedMemoryBuffer.Canvas.DrawImage(image, srcRect, dstRect, samplingOptions);
            }
            else
            {
                // Calculate the new image size based on the aspect ratio
                float scaleFactor = Math.Min((float)modelWidth / width, (float)modelHeight / height);

                // Use integer rounding instead of Math.Round
                int newWidth = (int)((width * scaleFactor) + 0.5f);
                int newHeight = (int)((height * scaleFactor) + 0.5f);

                // Calculate the destination rectangle within the model dimensions
                int x = (modelWidth - newWidth) / 2;
                int y = (modelHeight - newHeight) / 2;

                var srcRect = new SKRect(0, 0, width, height);
                var dstRect = new SKRect(x, y, x + newWidth, y + newHeight);

                // Draw the resized image onto the pinned memory buffer canvas as RGB888x with padding
                pinnedMemoryBuffer.Canvas.DrawImage(image, srcRect, dstRect, samplingOptions);
            }

            // Only dispose if we created a new SKImage from SKBitmap or if we cropped to a ROI, since cropping creates a new SKImage instance
            if (createdImage || roi.HasValue)
                image?.Dispose();

            // Return the input image dimensions (ROI dimensions if cropped), required to correctly scale bounding boxes
            return new SKSizeI(width, height);
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
