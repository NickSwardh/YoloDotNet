using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.Fonts;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.Drawing;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System.Globalization;
using System.Text;
using YoloDotNet.Enums;
using YoloDotNet.Models;

namespace YoloDotNet.Extensions
{
    public static class ImageExtension
    {
        /// <summary>
        /// Creates a resized clone of the input image with new width and height.
        /// </summary>
        /// <param name="image">The original image to be resized.</param>
        /// <param name="w">The desired width for the resized image.</param>
        /// <param name="h">The desired height for the resized image.</param>
        /// <returns>A new image with the specified dimensions.</returns>
        public static Image<Rgb24> ResizeImage(this Image image, int w, int h)
            => image.Clone(x => x.Resize(w, h)).CloneAs<Rgb24>();

        /// <summary>
        /// Extracts pixel values from an RGB image and converts them into a tensor.
        /// </summary>
        /// <param name="img">The RGB image to extract pixel values from.</param>
        /// <returns>A tensor containing normalized pixel values extracted from the input image.</returns>
        public static Tensor<float> ExtractPixelsFromImage(this Image<Rgb24> img, int inputBatchSize, int inputChannels)
        {
            var (width, height) = (img.Width, img.Height);

            var tensor = new DenseTensor<float>(new[] { inputBatchSize, inputChannels, width, height });

            Parallel.For(0, height, y =>
            {
                var pixelSpan = img.DangerousGetPixelRowMemory(y).Span;

                for (int x = 0; x < width; x++)
                {
                    tensor[0, 0, y, x] = pixelSpan[x].R / 255.0F; // r
                    tensor[0, 1, y, x] = pixelSpan[x].G / 255.0F; // g
                    tensor[0, 2, y, x] = pixelSpan[x].B / 255.0F; // b
                }
            });

            return tensor;
        }

        /// <summary>
        /// Draws bounding boxes and associated labels on the image.
        /// </summary>
        /// <param name="image">The image on which to draw the boxes and labels.</param>
        /// <param name="predictions">The collection of prediction results containing bounding boxes and labels.</param>
        public static void DrawBoundingBoxes(this Image image, IEnumerable<ObjectDetection> labels, bool drawConfidence = true)
        {
            // Define constants for readability
            const int fontSize = 16;
            const int borderWidth = 2;
            const int shadowOffset = 1;

            // Define fonts and colors
            var font = SystemFonts.Get(nameof(FontType.Arial))
                .CreateFont(fontSize, FontStyle.Bold);

            var shadowColor = new Rgba32(44, 44, 44, 180);
            var foregroundColor = new Rgba32(248, 240, 227, 224);

            image.Mutate(context =>
            {
                foreach (var label in labels!)
                {
                    var labelColor = HexToRgba(label.Label.Color, 128);

                    // Text with label name and confidence in percent
                    var text = label.Label.Name;

                    if (drawConfidence)
                    {
                        var confidencePercent = (pred!.Confidence * 100).ToString("0.##", CultureInfo.InvariantCulture);
                        text += $" ({confidencePercent}%)";
                    }

                    // Calculate text width and height
                    var textSize = TextMeasurer.MeasureSize(text, new TextOptions(font));

                    // Label x, y coordinates
                    var (x, y) = (label.Rectangle.X, label.Rectangle.Y - (textSize.Height * 2));

                    // Draw box
                    context.Draw(Pens.Solid(labelColor, borderWidth), label.Rectangle);

                    // Draw text background
                    context.Fill(labelColor, new RectangularPolygon(x, y, textSize.Width + fontSize, textSize.Height * 2));

                    // Draw text shadow
                    context.DrawText(text, font, shadowColor, new PointF(x + shadowOffset + (fontSize / 2), y + shadowOffset + (textSize.Height / 2)));

                    // Draw label text
                    context.DrawText(text, font, foregroundColor, new PointF(x + (fontSize / 2), y + (textSize.Height / 2)));
                }
            });
        }

        public static void DrawClassificationLabels(this Image image, IEnumerable<Classification>? labels, bool drawConfidence = true)
        {
            // Define constants for readability
            const int fontSize = 16;
            const int x = fontSize;
            const int y = fontSize;
            const int margin = fontSize / 2;
            const float lineSpace = 1.5f;

            // Define fonts and colors
            var font = GetFont(fontSize);
            var shadowColor = new Rgba32(0, 0, 0, 60);
            var foregroundColor = new Rgba32(255, 255, 255);

            var options = new RichTextOptions(font)
            {
                LineSpacing = lineSpace,
                Origin = new PointF(x + margin, y + margin)
            };

            // Gather labels and confidence score
            var sb = new StringBuilder();
            foreach (var label in labels!)
            {
                var text = label.Label;

                sb.AppendLine(text);
            }

            image.Mutate(context =>
            {
                // Calculate text width and height
                var textSize = TextMeasurer.MeasureSize(sb.ToString(), options);

                // Draw background
                context.Fill(shadowColor, new RectangularPolygon(x, y, textSize.Width + fontSize, textSize.Height + fontSize));

                // Draw labels
                context.DrawText(options, sb.ToString(), foregroundColor);
            });
        }

        private static Font GetFont(int size)
            => SystemFonts.Get(nameof(FontType.Arial))
                .CreateFont(size, FontStyle.Bold);

        /// <summary>
        /// Converts a hexadecimal color representation to an Rgba32 color.
        /// </summary>
        /// <param name="hexColor">The hexadecimal color value (e.g., "#RRGGBB").</param>
        /// <param name="alpha">Optional. The alpha (transparency) value for the Rgba32 color (0-255, default is 255).</param>
        /// <returns>An Rgba32 color representation.</returns>
        /// <exception cref="ArgumentException">Thrown when the input hex color format is invalid.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when the alpha value is outside the valid range (0-255).</exception>
        private static Rgba32 HexToRgba(string hexColor, int alpha = 255)
        {
            var hexValid = Color.TryParseHex(hexColor, out _);

            if (hexColor.Length != 7 || hexValid is false)
                throw new ArgumentException("Invalid hexadecimal color format.");

            if (alpha < 0 || alpha > 255)
                throw new ArgumentOutOfRangeException("Alfa value must be between 0-255.");

            byte r = byte.Parse(hexColor.Substring(1, 2), NumberStyles.HexNumber);
            byte g = byte.Parse(hexColor.Substring(3, 2), NumberStyles.HexNumber);
            byte b = byte.Parse(hexColor.Substring(5, 2), NumberStyles.HexNumber);

            return new Rgba32(r, g, b, (byte)alpha);
        }

    }
}
