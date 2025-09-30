// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Benchmarks.Setup
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

            if (img is SKImage skImage)
                image = skImage;
            else if (img is SKBitmap skBitmap)
                image = SKImage.FromPixels(skBitmap.Info, skBitmap.GetPixels());

            int modelWidth = pinnedMemoryBuffer.ImageInfo.Width;
            int modelHeight = pinnedMemoryBuffer.ImageInfo.Height;

            var srcRect = new SKRect(0, 0, image.Width, image.Height);
            var destRect = new SKRect(0, 0, modelWidth, modelHeight);

            pinnedMemoryBuffer.Canvas.DrawImage(image, srcRect, destRect, samplingOptions);

            // Return the original image dimensions, which are required to correctly scale bounding boxes
            return new SKSizeI(image.Width, image.Height);
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

            if (img is SKImage skImage)
            {
                image = skImage;
            }
            else if (img is SKBitmap skBitmap)
            {
                image = SKImage.FromPixels(skBitmap.Info, skBitmap.GetPixels());
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

            // Return the original image dimensions, which are required to correctly scale bounding boxes
            return new SKSizeI(width, height);
        }

        //private static bool IsImageCompatibleWithTargetInfo(SKBitmap image, SKImageInfo skInfo)
        //{
        //    bool sizeMatches = image.Width == skInfo.Width && image.Height == skInfo.Height;

        //    bool colorMatches = image.ColorType == skInfo.ColorType &&
        //                        image.AlphaType == skInfo.AlphaType &&
        //                        (image.ColorSpace?.Equals(skInfo.ColorSpace) ?? image.ColorSpace == null);

        //    return sizeMatches && colorMatches;
        //}

        /// <summary>
        /// Converts the pixel values of a given image into a normalized DenseTensor object.
        /// </summary>
        /// <param name="pixelsPtr">A pointer to the raw pixel image data in memory.</param>
        /// <param name="inputShape">The shape of the input tensor.</param>
        /// <param name="tensorBufferSize">The size of the tensor buffer, which should be equal to the product of the input shape dimensions.</param>
        /// <param name="tensorArrayBuffer">A pre-allocated float array buffer to store the normalized pixel values.</param>
        /// <returns>A DenseTensor&lt;float&gt; object containing normalized pixel values from the input image, arranged according to the specified input shape.</returns>
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
            byte* pixels = (byte*)pixelsPtr;

            // Normalize gray-scale or RGB pixel data into the tensor array buffer.
            if (colorChannels == 1)
            {
                float* dst = (float*)Unsafe.AsPointer(ref tensorArrayBuffer[0]);

                int i = 0;
                int limit = totalPixels & ~3;

                for (; i < limit; i += 4)
                {
                    dst[i] = pixels[i] * inv255;
                    dst[i + 1] = pixels[i + 1] * inv255;
                    dst[i + 2] = pixels[i + 2] * inv255;
                    dst[i + 3] = pixels[i + 3] * inv255;
                }
            }
            // Normalize RGB pixel data into the tensor array buffer.
            else
            {
                float* dstR = (float*)Unsafe.AsPointer(ref tensorArrayBuffer[0]);
                float* dstG = dstR + totalPixels;
                float* dstB = dstG + totalPixels;

                int i = 0;
                int limit = totalPixels & ~1; // Round down to nearest multiple of 16

                for (; i < limit; i += 2)
                {
                    dstR[i + 0] = pixels[0] * inv255;
                    dstG[i + 0] = pixels[1] * inv255;
                    dstB[i + 0] = pixels[2] * inv255;

                    dstR[i + 1] = pixels[4] * inv255;
                    dstG[i + 1] = pixels[5] * inv255;
                    dstB[i + 1] = pixels[6] * inv255;

                    pixels += 8;
                }
            }
        }


    }
}
