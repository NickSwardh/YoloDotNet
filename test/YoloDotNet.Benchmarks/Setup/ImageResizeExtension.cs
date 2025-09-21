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
        unsafe public static void NormalizePixelsToTensor(this IntPtr pixelsPtr,
            long[] inputShape,
            int tensorBufferSize,
            float[] tensorArrayBuffer)
        {
            // Deconstruct the input shape into batch size, number of channels, width, and height.
            var (batchSize, colorChannels, width, height) = ((int)inputShape[0], (int)inputShape[1], (int)inputShape[2], (int)inputShape[3]);

            // Total number of pixels in the image.
            int totalPixels = width * height;

            // Each color channel occupies a contiguous section in the tensor buffer.
            int pixelsPerChannel = tensorBufferSize / colorChannels;

            // Precompute the inverse multiplier constant for normalizing byte values (0-255) to the [0, 1] range.
            // This value (1.0f / 255.0f) is a quick way to convert any byte color component into a float between 0 and 1.
            // For example: a red component with value 128 becomes 128 * inv255 = 128 / 255 = 0.50196.
            float inv255 = 1.0f / 255.0f;

            // Lock the pixel data for fast, unsafe memory access.
            byte* pixels = (byte*)pixelsPtr;

            // Loop through all pixels in the image.
            for (int i = 0; i < totalPixels; i++)
            {
                // Compute the offset into the pixel array.
                int offset = i * 4;  // Assuming pixel format is RGBx or similar with 4 bytes per pixel.

                // Read the red, green, and blue components.
                byte* px = pixels + offset;
                byte r = px[0];
                byte g = px[1];
                byte b = px[2];

                // If the pixel is completely black, skip it.
                if ((r | g | b) == 0)
                    continue;

                // Normalize the red, green, and blue components and store them in the buffer.
                // The buffer is arranged in "channel-first" order:
                // - Red values go in the first section (0 to pixelsPerChannel)
                // - Green values go in the second section (pixelsPerChannel to 2 * pixelsPerChannel)
                // - Blue values go in the third section (2 * pixelsPerChannel to 3 * pixelsPerChannel)
                tensorArrayBuffer[i] = r * inv255;
                tensorArrayBuffer[i + pixelsPerChannel] = g * inv255;
                tensorArrayBuffer[i + 2 * pixelsPerChannel] = b * inv255;
            }
        }


    }
}
