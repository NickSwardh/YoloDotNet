// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Extensions
{
    internal static class ImageResizeExtension
    {
        /// <summary>
        /// Resizes the input image to the target dimensions by stretching it to fit the model input size, returning a pointer to RGB888x pixel data and the new dimensions.
        /// </summary>
        /// <param name="img">The original image to resize.</param>
        /// <param name="samplingOptions">Sampling options used during resizing.</param>
        /// <param name="pinnedMemoryBuffer">A pinned memory buffer where the resized image will be written.</param>
        /// <returns>A tuple containing a pointer to the resized image data and its dimensions.</returns>
        public static SKSizeI ResizeImageStretched<T>(this T img, SKSamplingOptions samplingOptions, PinnedMemoryBuffer pinnedMemoryBuffer)
        {
            SKImage image = default!;
            var createdImage = false;

            if (img is SKImage skImage)
                image = skImage;
            else if (img is SKBitmap skBitmap)
            {
                image = SKImage.FromPixels(skBitmap.Info, skBitmap.GetPixels());
                createdImage = true;
            }

            int modelWidth = pinnedMemoryBuffer.ImageInfo.Width;
            int modelHeight = pinnedMemoryBuffer.ImageInfo.Height;

            var srcRect = new SKRect(0, 0, image.Width, image.Height);
            var destRect = new SKRect(0, 0, modelWidth, modelHeight);

            pinnedMemoryBuffer.Canvas.DrawImage(image, srcRect, destRect, samplingOptions);
            var w = image.Width;
            var h = image.Height;

            // Only dispose if we created a bew SKImage from SKBitmap
            if (createdImage)
                image?.Dispose();

            // Return the original image dimensions, which are required to correctly scale bounding boxes
            return new SKSizeI(w, h);
        }

        /// <summary>
        /// Resizes the input image proportionally to fit the model input size, with RGB888x format and padded borders, returning a pointer to the pixel data and the new image dimensions.
        /// </summary>
        /// <param name="img">The original image to resize.</param>
        /// <param name="samplingOptions">Sampling options used during resizing.</param>
        /// <param name="pinnedMemoryBuffer">A pinned memory buffer where the resized image will be written.</param>
        /// <returns>A tuple containing a pointer to the resized image data and its dimensions.</returns>
        public static SKSizeI ResizeImageProportional<T>(this T img, SKSamplingOptions samplingOptions, PinnedMemoryBuffer pinnedMemoryBuffer)
        {
            SKImage image = default!;
            var createdImage = false;

            if (img is SKImage skImage)
                image = skImage;
            else if (img is SKBitmap skBitmap)
            {
                image = SKImage.FromPixels(skBitmap.Info, skBitmap.GetPixels());
                createdImage = true;
            }

            int modelWidth = pinnedMemoryBuffer.ImageInfo.Width;
            int modelHeight = pinnedMemoryBuffer.ImageInfo.Height;
            int width = image.Width;
            int height = image.Height;

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

            pinnedMemoryBuffer.Canvas.DrawImage(image, srcRect, dstRect, samplingOptions);
            var w = image.Width;
            var h = image.Height;

            // Only dispose if we created a bew SKImage from SKBitmap
            if (createdImage)
                image?.Dispose();

            // Return the original image dimensions, which are required to correctly scale bounding boxes
            return new SKSizeI(w, h);
        }

        /// <summary>
        /// Converts raw pixel image data to a normalized float array for model input.
        /// </summary>
        /// <param name="pixelsPtr">A pointer to the raw pixel image data in memory.</param>
        /// <param name="inputShape">The shape of the input tensor.</param>
        /// <param name="tensorBufferSize">The size of the tensor buffer, which should be equal to the product of the input shape dimensions.</param>
        /// <param name="tensorArrayBuffer">A pre-allocated float array buffer to store the normalized pixel values.</param>
        unsafe public static void NormalizePixelsToArray(this IntPtr pixelsPtr,
            long[] inputShape,
            int tensorBufferSize,
            float[] tensorArrayBuffer)
        {
            var colorChannels = (int)inputShape[1];
            var height = (int)inputShape[2];
            var width = (int)inputShape[3];
            int totalPixels = width * height;

            // Precompute the inverse multiplier constant for normalizing byte values (0-255) to the [0, 1] range.
            // This value (1.0f / 255.0f) is a quick way to convert any byte color component into a float between 0 and 1.
            // For example: a red component with value 128 becomes 128 * inv255 = 128 / 255 = 0.50196.
            float inv255 = 1.0f / 255.0f;

            // Lock the pixel data for fast, unsafe memory access.
            byte* pixels = (byte*)pixelsPtr;

            // Normalize gray-scale pixel data
            if (colorChannels == 1)
            {
                float* dst = (float*)Unsafe.AsPointer(ref tensorArrayBuffer[0]);

                int i = 0;
                int limit = totalPixels & ~3; // Round down to nearest multiple of 4

                // Unroll loop to process 4 pixels at a time for better performance
                for (; i < limit; i += 4)
                {
                    dst[i] = pixels[i] * inv255;
                    dst[i + 1] = pixels[i + 1] * inv255;
                    dst[i + 2] = pixels[i + 2] * inv255;
                    dst[i + 3] = pixels[i + 3] * inv255;
                }
            }
            // Normalize RGB pixel data
            else
            {
                float* dstR = (float*)Unsafe.AsPointer(ref tensorArrayBuffer[0]);
                float* dstG = dstR + totalPixels;
                float* dstB = dstG + totalPixels;

                int i = 0;
                int limit = totalPixels & ~1; // Round down to nearest multiple of 2

                // Unroll loop to process 2 pixels at a time for better performance
                for (; i < limit; i += 2)
                {
                    // Pixel 1
                    dstR[i + 0] = pixels[0] * inv255; // Red
                    dstG[i + 0] = pixels[1] * inv255; // Green
                    dstB[i + 0] = pixels[2] * inv255; // Blue
                    // Skip Alpha channel (pixels[3])

                    // Pixel 2
                    dstR[i + 1] = pixels[4] * inv255; // Red
                    dstG[i + 1] = pixels[5] * inv255; // Green
                    dstB[i + 1] = pixels[6] * inv255; // Blue
                    // Skip Alpha channel (pixels[7])

                    // Move to the next two pixels (8 bytes)
                    pixels += 8;
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
        unsafe public static void NormalizePixelsToArray(this IntPtr pixelsPtr,
            long[] inputShape,
            int tensorBufferSize,
            ushort[] tensorArrayBuffer)
        {
            var colorChannels = (int)inputShape[1];
            var height = (int)inputShape[2];
            var width = (int)inputShape[3];
            int totalPixels = width * height;

            // Precompute the inverse multiplier constant for normalizing byte values (0-255) to the [0, 1] range.
            // This value (1.0f / 255.0f) is a quick way to convert any byte color component into a float between 0 and 1.
            // For example: a red component with value 128 becomes 128 * inv255 = 128 / 255 = 0.50196.
            float inv255 = 1.0f / 255.0f;

            // Lock the pixel data for fast, unsafe memory access.
            byte* pixels = (byte*)pixelsPtr;

            // Normalize gray-scale pixel data
            if (colorChannels == 1)
            {
                ushort* dst = (ushort*)Unsafe.AsPointer(ref tensorArrayBuffer[0]);

                int i = 0;
                int limit = totalPixels & ~3; // Round down to nearest multiple of 4

                // Unroll loop to process 4 pixels at a time for better performance
                for (; i < limit; i += 4)
                {
                    dst[i] = FloatToUshort(pixels[i] * inv255);
                    dst[i + 1] = FloatToUshort(pixels[i + 1] * inv255);
                    dst[i + 2] = FloatToUshort(pixels[i + 2] * inv255);
                    dst[i + 3] = FloatToUshort(pixels[i + 3] * inv255);
                }
            }
            // Normalize RGB pixel data
            else
            {
                ushort* dstR = (ushort*)Unsafe.AsPointer(ref tensorArrayBuffer[0]);
                ushort* dstG = dstR + totalPixels;
                ushort* dstB = dstG + totalPixels;

                int i = 0;
                int limit = totalPixels & ~1; // Round down to nearest multiple of 2

                // Unroll loop to process 2 pixels at a time for better performance
                for (; i < limit; i += 2)
                {
                    // Pixel 1
                    dstR[i + 0] = FloatToUshort(pixels[0] * inv255); // Red
                    dstG[i + 0] = FloatToUshort(pixels[1] * inv255); // Green
                    dstB[i + 0] = FloatToUshort(pixels[2] * inv255); // Blue
                    // Skip Alpha channel (pixels[3])

                    // Pixel 2
                    dstR[i + 1] = FloatToUshort(pixels[4] * inv255); // Red
                    dstG[i + 1] = FloatToUshort(pixels[5] * inv255); // Green
                    dstB[i + 1] = FloatToUshort(pixels[6] * inv255); // Blue
                    // Skip Alpha channel (pixels[7])

                    // Move to the next two pixels (8 bytes)
                    pixels += 8;
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
