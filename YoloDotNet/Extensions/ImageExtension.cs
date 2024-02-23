namespace YoloDotNet.Extensions
{
    public static class ImageExtension
    {
        /// <summary>
        /// Draw classification labels on the given image, optionally including confidence scores.
        /// </summary>
        /// <param name="image">The image on which the labels are to be drawn.</param>
        /// <param name="detections">An collection of classification labels and confidence scores.</param>
        /// <param name="drawConfidence">A flag indicating whether to include confidence scores in the labels.</param>
        public static void Draw(this Image image, IEnumerable<Classification>? classifications, bool drawConfidence = true)
            => image.DrawClassificationLabels(classifications, drawConfidence);

        /// <summary>
        /// Draw bounding boxes around detected objects on the specified image.
        /// </summary>
        /// <param name="image">The image on which to draw bounding boxes.</param>
        /// <param name="detections">An enumerable collection of objects representing the detected items.</param>
        /// <param name="drawConfidence">A boolean indicating whether to include confidence percentages in the drawn labels.</param>
        public static void Draw(this Image image, IEnumerable<ObjectDetection>? detections, bool drawConfidence = true)
            => image.DrawBoundingBoxes(detections, drawConfidence);

        /// <summary>
        /// Draw segmentations and bounding boxes on the specified image.
        /// </summary>
        /// <param name="image">The image on which to draw segmentations.</param>
        /// <param name="detections">A list of segmentation information, including rectangles and segmented pixels.</param>
        /// <param name="drawConfidence">A boolean indicating whether to include confidence percentages in the drawn bounding boxes.</param>
        public static void Draw(this Image image, IEnumerable<Segmentation>? segmentations, DrawSegment draw = DrawSegment.Default, bool drawConfidence = true)
            => image.DrawSegmentations(segmentations, draw, drawConfidence);

        /// <summary>
        /// Creates a resized clone of the input image with new width, height and padded borders to fit new size.
        /// </summary>
        /// <param name="image">The original image to be resized.</param>
        /// <param name="w">The desired width for the resized image.</param>
        /// <param name="h">The desired height for the resized image.</param>
        /// <returns>A new image with the specified dimensions.</returns>
        public static Image<Rgb24> ResizeImage(this Image image, int w, int h)
        {
            var options = new ResizeOptions
            {
                Size = new Size(w, h),
                Mode = ResizeMode.Pad,
                PadColor = new Color(new Rgb24(0, 0, 0))
            };

            return image.Clone(x => x.Resize(options)).CloneAs<Rgb24>();
        }

        /// <summary>
        /// Extracts pixel values from an image and converts them into a tensor.
        /// </summary>
        /// <param name="img">The image to extract pixel values from.</param>
        /// <returns>A tensor containing normalized pixel values extracted from the input image.</returns>
        public static Tensor<float> PixelsToTensor(this Image<Rgb24> img, int inputBatchSize, int inputChannels)
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
        /// Retrieves segmented pixels from an image based on the specified function.
        /// </summary>
        /// <param name="image">The image from which to retrieve segmented pixels.</param>
        /// <param name="func">A function that computes confidence values for pixels.</param>
        /// <returns>An array of <see cref="Pixel"/> representing the segmented pixels.</returns>
        public static Pixel[] GetSegmentedPixels(this Image<L8> image, Func<L8, float> func)// where L8 : unmanaged, IPixel<L8>
        {
            var width = image.Width;
            var height = image.Height;

            var pixels = new ConcurrentBag<Pixel>();

            Parallel.For(0, height, y =>
            {
                var row = image.DangerousGetPixelRowMemory(y).Span;

                for (int x = 0; x < width; x++)
                {
                    var confidence = func(row[x]);

                    if (confidence > 0.68f)
                        pixels.Add(new Pixel(x, y, confidence));
                }
            });

            return pixels.ToArray();
        }

        /// <summary>
        /// Resizes a segmented image to the original image size and crops segmented area.
        /// </summary>
        /// <param name="image">The segmented image to be resized and cropped.</param>
        /// <param name="originalImage">The original image used as a reference for resizing.</param>
        /// <param name="rectangle">The rectangle specifying the area to be cropped after resizing.</param>
        public static void CropResizedSegmentedArea(this Image image, Image originalImage, Rectangle rectangle)
        {
            var gain = Math.Min(image.Width / (float)originalImage.Width, image.Height / (float)originalImage.Height);

            var x = (int)((image.Width - originalImage.Width * gain) / 2);
            var y = (int)((image.Height - originalImage.Height * gain) / 2);
            var w = image.Width - x * 2;
            var h = image.Height - y * 2;

            image.Mutate(img =>
            {
                img.Crop(new Rectangle(x, y, w, h));
                img.Resize(originalImage.Width, originalImage.Height);
                img.Crop(rectangle);
            });
        }

        #region Helper methods

        /// <summary>
        /// Helper method for drawing classification labels.
        /// </summary>
        /// <param name="image">The image on which the labels are to be drawn.</param>
        /// <param name="labels">An collection of classification labels and confidence scores.</param>
        /// <param name="drawConfidence">A flag indicating whether to include confidence scores in the labels.</param>
        private static void DrawClassificationLabels(this Image image, IEnumerable<Classification>? labels, bool drawConfidence = true)
        {
            ArgumentNullException.ThrowIfNull(labels);

            var fontSize = image.CalculateFontSizeByDpi(16f);
            var x = (int)fontSize;
            var y = (int)fontSize;
            var margin = fontSize / 2;
            var lineSpace = 1.5f;

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

                if (drawConfidence)
                    text += $" ({label!.Confidence.ToPercent()}%)";

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

        /// <summary>
        /// Helper method for drawing segmentations and bounding boxes.
        /// </summary>
        /// <param name="image">The image on which to draw segmentations.</param>
        /// <param name="segmentations">A list of segmentation information, including rectangles and segmented pixels.</param>
        /// <param name="drawConfidence">A boolean indicating whether to include confidence percentages in the drawn bounding boxes.</param>
        private static void DrawSegmentations(this Image image, IEnumerable<Segmentation>? segmentations, DrawSegment draw, bool drawConfidence)
        {
            ArgumentNullException.ThrowIfNull(segmentations);

            if (draw == DrawSegment.Default || draw == DrawSegment.PixelMaskOnly)
            {
                var options = new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount };

                Parallel.ForEach(segmentations, options, segmentation =>
                {
                    // Create a new transparent image
                    using var mask = new Image<Rgba32>(segmentation.BoundingBox.Width, segmentation.BoundingBox.Height);

                    var color = Color.ParseHex(segmentation.Label.Color);

                    // Add color to segmented pixels
                    var test = segmentation.SegmentedPixels.AsSpan();
                    foreach (var pixel in test)
                        mask[pixel.X, pixel.Y] = color;

                    image.Mutate(x => x.DrawImage(mask, segmentation.BoundingBox.Location, .38f));
                });
            }

            if (draw == DrawSegment.Default || draw == DrawSegment.BoundingBoxOnly)
                image.DrawBoundingBoxes(segmentations, drawConfidence);
        }

        /// <summary>
        /// Helper method for drawing bounding boxes around detected objects.
        /// </summary>
        /// <param name="image">The image on which to draw bounding boxes.</param>
        /// <param name="detections">An enumerable collection of objects representing the detected items.</param>
        /// <param name="drawConfidence">A boolean indicating whether to include confidence percentages in the drawn labels.</param>
        private static void DrawBoundingBoxes(this Image image, IEnumerable<IDetection>? detections, bool drawConfidence)
        {
            ArgumentNullException.ThrowIfNull(detections);

            // Define constants for readability
            const int borderThickness = 2;
            const int shadowOffset = 1;

            // Define fonts and colors
            var fontSize = image.CalculateFontSizeByDpi(16f);
            var font = SystemFonts.Get(nameof(FontType.Arial))
                .CreateFont(fontSize, FontStyle.Bold);

            var shadowColor = new Rgba32(44, 44, 44, 180);
            var foregroundColor = new Rgba32(248, 240, 227, 224);

            image.Mutate(context =>
            {
                foreach (var label in detections!)
                {
                    var labelColor = HexToRgba(label.Label.Color, 128);

                    // Text with label name and confidence in percent
                    var text = label.Label.Name;

                    if (drawConfidence)
                        text += $" ({label!.Confidence.ToPercent()}%)";

                    // Calculate text width and height
                    var textSize = TextMeasurer.MeasureSize(text, new TextOptions(font));

                    // Label x, y coordinates
                    var (x, y) = (label.BoundingBox.X, label.BoundingBox.Y - (textSize.Height * 2));

                    // Draw box
                    context.Draw(Pens.Solid(labelColor, borderThickness), label.BoundingBox);

                    // Draw text background
                    context.Fill(labelColor, new RectangularPolygon(x, y, textSize.Width + fontSize, textSize.Height * 2));

                    // Draw text shadow
                    context.DrawText(text, font, shadowColor, new PointF(x + shadowOffset + (fontSize / 2), y + shadowOffset + (textSize.Height / 2)));

                    // Draw label text
                    context.DrawText(text, font, foregroundColor, new PointF(x + (fontSize / 2), y + (textSize.Height / 2)));
                }
            });
        }

        /// <summary>
        /// Gets the default label font
        /// </summary>
        /// <param name="size"></param>
        /// <returns></returns>
        private static Font GetFont(float size)
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
                throw new ArgumentOutOfRangeException(nameof(alpha), "Alfa value must be between 0-255.");

            byte r = byte.Parse(hexColor.Substring(1, 2), NumberStyles.HexNumber);
            byte g = byte.Parse(hexColor.Substring(3, 2), NumberStyles.HexNumber);
            byte b = byte.Parse(hexColor.Substring(5, 2), NumberStyles.HexNumber);

            return new Rgba32(r, g, b, (byte)alpha);
        }

        #endregion

    }
}
