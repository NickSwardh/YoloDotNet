// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Utils
{
    /// <summary>
    /// Rasterizes segmentation mask outlines (closed polygons in image coordinates, filled with the even-odd rule).
    /// Without anti-aliasing a pixel belongs to the mask when its centre lies inside the outline.
    /// </summary>
    internal static unsafe class MaskRasterizer
    {
        /// <summary>
        /// Creates a bit-packed mask covering <paramref name="box"/>: 1 bit per pixel, row by row, least significant
        /// bit first.
        /// </summary>
        public static byte[] ToBitPacked(SKPoint[][] contours, SKRectI box)
        {
            int width = box.Width, height = box.Height;
            int totalPixels = width * height;

            if (width <= 0 || height <= 0)
                return [];

            var bits = new byte[(totalPixels + 7) / 8];

            if (contours.Length == 0)
                return bits;

            var buffer = ArrayPool<byte>.Shared.Rent(totalPixels);

            try
            {
                Array.Clear(buffer, 0, totalPixels);

                fixed (byte* pixels = buffer)
                {
                    // Row bytes = width, so pixel i of the box is byte i of the buffer, as in the packed mask.
                    Fill(pixels, width, height, width, contours, box.Left, box.Top, antiAlias: false);
                    Pack(pixels, totalPixels, bits);
                }
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(buffer);
            }

            return bits;
        }

        /// <summary>
        /// Creates a Gray8 mask bitmap (0 = background, 255 = mask; partial values along the edge when anti-aliased)
        /// whose top-left pixel is at (<paramref name="left"/>, <paramref name="top"/>) in image coordinates.
        /// </summary>
        public static SKBitmap ToBitmap(SKPoint[][] contours, int left, int top, int width, int height, bool antiAlias)
        {
            var bitmap = new SKBitmap(new SKImageInfo(Math.Max(1, width), Math.Max(1, height), SKColorType.Gray8, SKAlphaType.Opaque));
            bitmap.Erase(SKColors.Black);

            if (width > 0 && height > 0 && contours.Length > 0)
                Fill((byte*)bitmap.GetPixels(), width, height, bitmap.RowBytes, contours, left, top, antiAlias);

            return bitmap;
        }

        /// <summary>
        /// Fills the outlines into one byte per pixel (coverage, 0-255) through an Alpha8 view of the memory.
        /// </summary>
        private static void Fill(byte* pixels, int width, int height, int rowBytes, SKPoint[][] contours, float left, float top, bool antiAlias)
        {
            using var bitmap = new SKBitmap();
            bitmap.InstallPixels(new SKImageInfo(width, height, SKColorType.Alpha8, SKAlphaType.Premul), (IntPtr)pixels, rowBytes);

            using var canvas = new SKCanvas(bitmap);
            using var path = CreatePath(contours);
            using var paint = new SKPaint { Color = SKColors.Black, IsAntialias = antiAlias, Style = SKPaintStyle.Fill };

            canvas.Translate(-left, -top);
            canvas.DrawPath(path, paint);
        }

        /// <summary>
        /// Builds an even-odd path from the outlines, so holes stay empty whatever the winding of each outline.
        /// </summary>
        public static SKPath CreatePath(SKPoint[][] contours)
        {
            var path = new SKPath { FillType = SKPathFillType.EvenOdd };

            foreach (var contour in contours)
                path.AddPoly(contour, close: true);

            return path;
        }

        /// <summary>
        /// Packs one byte per pixel into one bit per pixel (set when the byte is above 127), 32 pixels per AVX2 step:
        /// MoveMask collects the top bit of each byte, lowest pixel in the lowest bit, which is the packed bit order.
        /// </summary>
        private static void Pack(byte* pixels, int totalPixels, byte[] bits)
        {
            int i = 0;

            fixed (byte* destination = bits)
            {
                if (Avx2.IsSupported)
                {
                    for (; i + 32 <= totalPixels; i += 32)
                        *(uint*)(destination + (i >> 3)) = (uint)Avx2.MoveMask(Avx.LoadVector256(pixels + i));
                }

                for (; i < totalPixels; i++)
                {
                    if (pixels[i] > 127)
                        destination[i >> 3] |= (byte)(1 << (i & 0b0111));
                }
            }
        }
    }
}
