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

            float inv255 = 1.0f / 255.0f;
            byte* src = (byte*)pixelsPtr;

            if (colorChannels == 1)
            {
                float* dst = (float*)Unsafe.AsPointer(ref tensorArrayBuffer[0]);
                int srcIndex = 0;

                for (int i = 0; i < totalPixels; i++, srcIndex += 4)
                {
                    // Read only the grayscale component (assumed in R channel)
                    dst[i] = src[srcIndex] * inv255;
            }
            }
            else
            {
                float* dstR = (float*)Unsafe.AsPointer(ref tensorArrayBuffer[0]);
                float* dstG = dstR + totalPixels;
                float* dstB = dstG + totalPixels;

                int srcIndex = 0;
                for (int i = 0; i < totalPixels; i++, srcIndex += 4)
        {
                    dstR[i] = src[srcIndex] * inv255;
                    dstG[i] = src[srcIndex + 1] * inv255;
                    dstB[i] = src[srcIndex + 2] * inv255;
            }
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

            float inv255 = 1.0f / 255.0f;
            byte* src = (byte*)pixelsPtr;

            if (colorChannels == 1)
            {
                ushort* dst = (ushort*)Unsafe.AsPointer(ref tensorArrayBuffer[0]);
                int srcIndex = 0;

                for (int i = 0; i < totalPixels; i++, srcIndex += 4)
                {
                    dst[i] = FloatToUshort(src[srcIndex] * inv255);
                }
            }
            else
            {
                ushort* dstR = (ushort*)Unsafe.AsPointer(ref tensorArrayBuffer[0]);
                ushort* dstG = dstR + totalPixels;
                ushort* dstB = dstG + totalPixels;

                int srcIndex = 0;
                for (int i = 0; i < totalPixels; i++, srcIndex += 4)
                {
                    dstR[i] = FloatToUshort(src[srcIndex] * inv255);
                    dstG[i] = FloatToUshort(src[srcIndex + 1] * inv255);
                    dstB[i] = FloatToUshort(src[srcIndex + 2] * inv255);
                }
            }
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
