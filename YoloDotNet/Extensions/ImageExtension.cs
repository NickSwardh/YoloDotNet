namespace YoloDotNet.Extensions
{
    public static class ImageExtension
    {
        /// <summary>
        /// Draw classification labels on the given image, optionally including confidence scores.
        /// </summary>
        /// <param name="image">The image on which the labels are to be drawn.</param>
        /// <param name="classifications">An collection of classification labels and confidence scores.</param>
        /// <param name="drawConfidence">A flag indicating whether to include confidence scores in the labels.</param>
        public static void Draw(this SKBitmap image,
            IEnumerable<Classification>? classifications,
            bool drawConfidence = true,
            SKTypeface font = default!)
            => image.DrawClassificationLabels(classifications, drawConfidence, font);

        /// <summary>
        /// Draw bounding boxes around detected objects on the specified image.
        /// </summary>
        /// <param name="image">The image on which to draw bounding boxes.</param>
        /// <param name="objectDetections">An enumerable collection of objects representing the detected items.</param>
        /// <param name="drawConfidence">A boolean indicating whether to include confidence percentages in the drawn labels.</param>
        public static void Draw(this SKBitmap image,
            IEnumerable<ObjectDetection>? objectDetections,
            bool drawConfidence = true,
            SKTypeface font = default!)
            => image.DrawBoundingBoxes(objectDetections, drawConfidence, font);

        /// <summary>
        /// Draw oriented bounding boxes around detected objects on the specified image.
        /// </summary>
        /// <param name="image">The image on which to draw oriented bounding boxes.</param>
        /// <param name="detections">An enumerable collection of objects representing the detected items.</param>
        /// <param name="drawConfidence">A boolean indicating whether to include confidence percentages in the drawn labels.</param>
        public static void Draw(this SKBitmap image,
            IEnumerable<OBBDetection>? detections,
            bool drawConfidence = true,
            SKTypeface font = default!)
            => image.DrawOrientedBoundingBoxes(detections, drawConfidence, font);

        /// <summary>
        /// Draw segmentations and bounding boxes on the specified image.
        /// </summary>
        /// <param name="image">The image on which to draw segmentations.</param>
        /// <param name="segmentations">A list of segmentation information, including rectangles and segmented pixels.</param>
        /// <param name="drawSegment">Specifies the segments to draw, with a default value.</param>
        /// <param name="drawConfidence">A boolean indicating whether to include confidence percentages in the drawn bounding boxes.</param>
        public static void Draw(this SKBitmap image,
            IEnumerable<Segmentation>? segmentations,
            DrawSegment drawSegment = DrawSegment.Default,
            bool drawConfidence = true,
            SKTypeface font = default!)
            => image.DrawSegmentations(segmentations, drawSegment, drawConfidence, font);

        /// <summary>
        /// Draws pose-estimated keypoints and bounding boxes on the specified image.
        /// </summary>
        /// <param name="image">The image on which to draw pose estimations.</param>
        /// <param name="poseEstimations">A list of pose estimations.</param>
        /// <param name="keyPointOptions">Options for drawing keypoints.</param>
        /// <param name="drawConfidence">A boolean indicating whether to include confidence percentages in the drawn bounding boxes.</param>
        public static void Draw(this SKBitmap image,
            IEnumerable<PoseEstimation>? poseEstimations,
            KeyPointOptions keyPointOptions,
            bool drawConfidence = true,
            SKTypeface font = default!)
            => image.DrawPoseEstimation(poseEstimations, keyPointOptions, drawConfidence, font);

        /// <summary>
        /// Saves the SKBitmap to a file with the specified format and quality.
        /// </summary>
        /// <param name="image">The SKBitmap to be saved.</param>
        /// <param name="filename">The name of the file where the image will be saved.</param>
        /// <param name="format">The format in which the image should be saved.</param>
        /// <param name="quality">The quality of the saved image (default is 100).</param>
        public static void Save(this SKBitmap image,
            string filename,
            SKEncodedImageFormat format = SKEncodedImageFormat.Jpeg,
            int quality = 100)
            => FrameSaveService.AddToQueue(image, filename, format, quality);

        ///// <summary>
        ///// Resizes the input image to fit the specified dimensions by stretching it, potentially distorting the aspect ratio.
        ///// The resulting image will have the specified width, height, and colorspace (RGB888x).
        ///// </summary>
        ///// <param name="image">The original image to be resized.</param>
        ///// <param name="skInfo">The desired SKImageInfo, including the target dimensions and colorspace.</param>
        ///// <returns>A new image stretched to fit the specified dimensions.</returns>
        public static nint ResizeImageStretched(this SKBitmap skbitmap, SKSamplingOptions samplingOptions, PinnedMemoryBuffer pinnedMemoryBuffer)
        {
            using var image = SKImage.FromPixels(skbitmap.Info, skbitmap.GetPixels());
            pinnedMemoryBuffer.Canvas.DrawImage(image, 0, 0, samplingOptions, ImageConfig.ResizePaint);

            return pinnedMemoryBuffer.Pointer;
        }

        ///// <summary>
        ///// Creates a resized proportional clone of the input image with new width, height, colorspace (RGB888x) and padded borders to fit the new size.
        ///// </summary>
        ///// <param name="image">The original image to be resized.</param>
        ///// <param name="skInfo">The desired SKImageInfo for the resized image.</param>
        ///// <returns>A new image with the specified dimensions.</returns>
        public static nint ResizeImageProportional(this SKBitmap skBitmapImage, SKSamplingOptions samplingOptions, PinnedMemoryBuffer pinnedMemoryBuffer)
        {
            using var image = SKImage.FromPixels(skBitmapImage.Info, skBitmapImage.GetPixels());

            int modelWidth = pinnedMemoryBuffer.ImageInfo.Width;
            int modelHeight = pinnedMemoryBuffer.ImageInfo.Height;
            int width = image.Width;
            int height = image.Height;

            // Calculate the new image size based on the aspect ratio
            float scaleFactor = Math.Min((float)modelWidth / width, (float)modelHeight / height);
            int newWidth = (int)Math.Round(width * scaleFactor); // Use integer rounding instead of Math.Round
            int newHeight = (int)Math.Round(height * scaleFactor);

            // Calculate the destination rectangle within the model dimensions
            int x = (modelWidth - newWidth) / 2;
            int y = (modelHeight - newHeight) / 2;

            var srcRect = new SKRect(0, 0, width, height);
            var dstRect = new SKRect(x, y, x + newWidth, y + newHeight);

            //buffer.Canvas.Clear(SKColors.Black);
            pinnedMemoryBuffer.Canvas.DrawImage(image, srcRect, dstRect, samplingOptions, ImageConfig.ResizePaint);
            //buffer.Canvas.Flush();

            return pinnedMemoryBuffer.Pointer;
        }

        private static bool IsImageCompatibleWithTargetInfo(SKBitmap image, SKImageInfo skInfo)
        {
            bool sizeMatches = image.Width == skInfo.Width && image.Height == skInfo.Height;

            bool colorMatches = image.ColorType == skInfo.ColorType &&
                                image.AlphaType == skInfo.AlphaType &&
                                (image.ColorSpace?.Equals(skInfo.ColorSpace) ?? image.ColorSpace == null);

            return sizeMatches && colorMatches;
        }

        ///// <summary>
        ///// Converts the pixel values of a given image into a normalized DenseTensor object.
        ///// </summary>
        ///// <param name="resizedImage">The image from which to extract pixel values.</param>
        ///// <param name="inputShape">The shape of the input tensor.</param>
        ///// <param name="tensorBufferSize">The size of the tensor buffer, which should be equal to the product of the input shape dimensions.</param>
        ///// <param name="tensorArrayBuffer">A pre-allocated float array buffer to store the normalized pixel values.</param>
        ///// <returns>A DenseTensor&lt;float&gt; object containing normalized pixel values from the input image, arranged according to the specified input shape.</returns>
        unsafe public static DenseTensor<float> NormalizePixelsToTensor(this nint imagePointer,
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
            IntPtr pixelsPtr = imagePointer;// resizedImage.GetPixels();
            byte* pixels = (byte*)pixelsPtr;
            
            // Use parallelism if image is larger than 224x224
            if (width > 224 && height > 244)
            {
                Parallel.For(0, totalPixels, i =>
                {
                    ComputePixels(pixels, i, pixelsPerChannel, inv255, tensorArrayBuffer);
                });
            }
            else
            {
                // Loop through all pixels in the image.
                for (int i = 0; i < totalPixels; i++)
                {
                    ComputePixels(pixels, i, pixelsPerChannel, inv255, tensorArrayBuffer);
                }
            }

            // Create and return a DenseTensor using the correctly sized memory slice.
            return new DenseTensor<float>(
                tensorArrayBuffer.AsMemory(0, tensorBufferSize),
                [batchSize, colorChannels, width, height]
            );
        }

        #region Helper methods

        unsafe private static void ComputePixels(byte* pixels, int index, int pixelsPerChannel, float inv255, float[] buffer)
        {
            // Compute the offset into the pixel array.
            int offset = index * 4;  // Assuming pixel format is RGBx or similar with 4 bytes per pixel.

            // Read the red, green, and blue components.
            byte r = pixels[offset];
            byte g = pixels[offset + 1];
            byte b = pixels[offset + 2];

            // Normalize the red, green, and blue components and store them in the buffer.
            // The buffer is arranged in "channel-first" order:
            // - Red values go in the first section (0 to pixelsPerChannel)
            // - Green values go in the second section (pixelsPerChannel to 2 * pixelsPerChannel)
            // - Blue values go in the third section (2 * pixelsPerChannel to 3 * pixelsPerChannel)
            buffer[index] = r * inv255;
            buffer[index + pixelsPerChannel] = g * inv255;
            buffer[index + 2 * pixelsPerChannel] = b * inv255;
        }

        /// <summary>
        /// Helper method for drawing classification labels.
        /// </summary>
        /// <param name="image">The image on which the labels are to be drawn.</param>
        /// <param name="labels">An collection of classification labels and confidence scores.</param>
        /// <param name="drawConfidence">A flag indicating whether to include confidence scores in the labels.</param>
        private static void DrawClassificationLabels(this SKBitmap image,
            IEnumerable<Classification>? labels,
            bool drawConfidence = true,
            SKTypeface fontType = default!)
        {
            ArgumentNullException.ThrowIfNull(labels);

            float x = ImageConfig.CLASSIFICATION_TRANSPARENT_BOX_X;
            float y = ImageConfig.CLASSIFICATION_TRANSPARENT_BOX_Y;

            using var font = new SKFont()
            {
                Size = ImageConfig.FONT_SIZE,
                Typeface = fontType ?? SKTypeface.Default
            };

            var fontSize = image.CalculateDynamicSize(font.Size);
            float margin = fontSize / 2;

            // Measure maximum text-length in order to determine the width of the transparent box
            float boxMaxWidth = 0;
            float boxMaxHeight = 0 - margin / 2;
            foreach (var label in labels)
            {
                var lineWidth = font.MeasureText(LabelText(label.Label, label.Confidence, drawConfidence));
                if (lineWidth > boxMaxWidth)
                    boxMaxWidth = lineWidth;

                boxMaxHeight += fontSize + margin;
            }

            using var canvas = new SKCanvas(image);

            // Draw transparent box for text
            canvas.DrawRect(SKRect.Create(x, y, boxMaxWidth + fontSize, boxMaxHeight + fontSize), ImageConfig.ClassificationBackgroundPaint);

            // Draw labels on transparent box
            y += font.Size;
            foreach (var label in labels!)
            {
                canvas.DrawText(LabelText(label.Label, label.Confidence, drawConfidence), x + margin, y + margin, font, ImageConfig.FontColorPaint);
                y += fontSize + margin;
            }
        }

        private static string LabelText(string labelName, double confidence, bool showConfidence)
        {
            var confidenceFormat = showConfidence ? $" ({confidence.ToPercent()}%)" : "";
            return $"{labelName}{confidenceFormat}";
        }

        /// <summary>
        /// Helper method for drawing segmentations and bounding boxes.
        /// </summary>
        /// <param name="image">The image on which to draw segmentations.</param>
        /// <param name="segmentations">A list of segmentation information, including rectangles and segmented pixels.</param>
        /// <param name="drawConfidence">A boolean indicating whether to include confidence percentages in the drawn bounding boxes.</param>
        unsafe private static void DrawSegmentations(this SKBitmap image,
            IEnumerable<Segmentation>? segmentations,
            DrawSegment draw,
            bool drawConfidence,
            SKTypeface fontType = default!)
        {
            ArgumentNullException.ThrowIfNull(segmentations);

            // Convert SKImage to SKBitmap to ensure pixel data is accessible
            // bitmap = SKBitmap.FromImage(image);
            IntPtr pixelsPtr = image.GetPixels();
            int width = image.Width;
            int height = image.Height;
            int bytesPerPixel = image.BytesPerPixel;
            int rowBytes = image.RowBytes;

            if (draw == DrawSegment.Default || draw == DrawSegment.PixelMaskOnly)
            {
                var options = new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount };

                Parallel.ForEach(segmentations, options, segmentation =>
                {
                    // Define the overlay color
                    var color = HexToRgbaSkia(segmentation.Label.Color, ImageConfig.SEGMENTATION_MASK_OPACITY);

                    var pixelSpan = segmentation.SegmentedPixels.AsSpan();

                    // Access pixel data directly from memory for higher performance
                    byte* pixelData = (byte*)pixelsPtr.ToPointer();

                    foreach (var pixel in pixelSpan)
                    {
                        int x = pixel.X;
                        int y = pixel.Y;

                        // Prevent any attempt to access or modify pixel data outside the valid range!
                        if (x < 0 || x >= width || y < 0 || y >= height)
                            continue;

                        // Calculate the index for the pixel
                        int index = y * rowBytes + x * bytesPerPixel;

                        // Get original pixel colors
                        byte blue = pixelData[index];
                        byte green = pixelData[index + 1];
                        byte red = pixelData[index + 2];
                        byte alpha = pixelData[index + 3];

                        // Blend the overlay color with the original color
                        byte newRed = (byte)((red * (255 - color.Alpha) + color.Red * color.Alpha) / 255);
                        byte newGreen = (byte)((green * (255 - color.Alpha) + color.Green * color.Alpha) / 255);
                        byte newBlue = (byte)((blue * (255 - color.Alpha) + color.Blue * color.Alpha) / 255);

                        // Set the new color
                        pixelData[index + 0] = newBlue;
                        pixelData[index + 1] = newGreen;
                        pixelData[index + 2] = newRed;
                        pixelData[index + 3] = alpha; // Preserve the original alpha
                    }
                });
            }

            if (draw == DrawSegment.Default || draw == DrawSegment.BoundingBoxOnly)
                image.DrawBoundingBoxes(segmentations, drawConfidence, fontType);
        }

        /// <summary>
        /// Helper method for drawing pose estimation and bounding boxes.
        /// </summary>
        /// <param name="image">The image on which to pose estimations.</param>
        /// <param name="poseEstimations">A list of pose estimation information, including rectangles and pose markers.</param>
        /// <param name="drawConfidence">A boolean indicating whether to include confidence percentages in the drawn bounding boxes.</param>
        private static void DrawPoseEstimation(this SKBitmap image,
            IEnumerable<PoseEstimation>? poseEstimations,
            KeyPointOptions poseOptions, bool drawConfidence,
            SKTypeface fontType = default!)
        {
            ArgumentNullException.ThrowIfNull(poseEstimations);

            var circleRadius = image.CalculateDynamicSize(ImageConfig.KEYPOINT_SIZE);
            var lineSize = image.CalculateDynamicSize(ImageConfig.BORDER_THICKNESS);
            var confidenceThreshold = poseOptions.PoseConfidence;
            var hasPoseMarkers = poseOptions.PoseMarkers.Length > 0;
            var emptyPoseMarker = new KeyPointMarker();
            var alpha = ImageConfig.DEFAULT_OPACITY;

            using var paint = new SKPaint() { Style = SKPaintStyle.Fill, IsAntialias = true };
            using var keyPointLinePaint = new SKPaint { StrokeWidth = lineSize, IsAntialias = true };
            using var canvas = new SKCanvas(image);

            foreach (var poseEstimation in poseEstimations)
            {
                var keyPoints = poseEstimation.KeyPoints.AsSpan();

                for (int i = 0; i < keyPoints.Length; i++)
                {
                    var keyPoint = keyPoints[i];

                    if (keyPoint.Confidence < confidenceThreshold)
                        continue;

                    var poseMap = hasPoseMarkers
                        ? poseOptions.PoseMarkers[i]
                        : emptyPoseMarker;

                    var color = hasPoseMarkers
                        ? HexToRgbaSkia(poseMap.Color, alpha)
                        : HexToRgbaSkia(poseOptions.DefaultPoseColor, alpha);

                    // Draw keypoint
                    paint.Color = color;
                    canvas.DrawCircle(keyPoint.X, keyPoint.Y, circleRadius, paint);

                    // Draw lines between key-points
                    foreach (var connection in poseMap.Connections)
                    {
                        var markerDestination = poseEstimation.KeyPoints[connection.Index];

                        if (markerDestination.Confidence < confidenceThreshold)
                            continue;

                        keyPointLinePaint.Color = HexToRgbaSkia(connection.Color, alpha);

                        canvas.DrawLine(
                             new SKPoint(keyPoint.X, keyPoint.Y),
                            new SKPoint(markerDestination.X, markerDestination.Y),
                            keyPointLinePaint);
                    }
                }
            }

            if (poseOptions.DrawBoundingBox)
                image.DrawBoundingBoxes(poseEstimations, drawConfidence, fontType);
        }

        /// <summary>
        /// Helper method for drawing bounding boxes around detected objects.
        /// </summary>
        /// <param name="image">The image on which to draw bounding boxes.</param>
        /// <param name="detections">An enumerable collection of objects representing the detected items.</param>
        /// <param name="drawConfidence">A boolean indicating whether to include confidence percentages in the drawn labels.</param>
        private static void DrawBoundingBoxes(this SKBitmap image,
            IEnumerable<IDetection>? detections,
            bool drawConfidence,
            SKTypeface fontType = default!)
        {
            ArgumentNullException.ThrowIfNull(detections);

            var fontSize = image.CalculateDynamicSize(ImageConfig.FONT_SIZE);
            var borderThickness = image.CalculateDynamicSize(ImageConfig.BORDER_THICKNESS);

            //float fontSize = image.CalculateFontSize(ImageConfig.DEFAULT_FONT_SIZE);
            var margin = (int)fontSize / 2;
            var labelBoxHeight = (int)fontSize * 2;
            var textOffset = (int)(fontSize + margin) - (margin / 2);
            var shadowOffset = ImageConfig.SHADOW_OFFSET;
            var labelOffset = (int)borderThickness / 2;
            byte labelBoxAlpha = ImageConfig.DEFAULT_OPACITY;

            using var font = new SKFont
            {
                Size = fontSize,
                Typeface = fontType ?? SKTypeface.Default
            };

            // Label box background paint
            using var labelBgPaint = new SKPaint
            {
                Style = SKPaintStyle.Fill,
                StrokeWidth = borderThickness
            };

            // Bounding box paint
            using var boxPaint = new SKPaint
            {
                Style = SKPaintStyle.Stroke,
                StrokeWidth = borderThickness
            };

            using var canvas = new SKCanvas(image);

            // Draw detections
            foreach (var detection in detections)
            {
                var box = detection.BoundingBox;
                var boxColor = HexToRgbaSkia(detection.Label.Color, labelBoxAlpha);

                var text = (detection.Id is not null)
                    ? $"Id: {detection.Id}, {detection.Label.Name}"
                    : detection.Label.Name;

                var labelText = LabelText(text, detection.Confidence, drawConfidence);
                var labelWidth = (int)font.MeasureText(labelText);

                labelBgPaint.Color = boxColor;
                boxPaint.Color = boxColor;

                // Calculate label background rect size
                var left = box.Left - labelOffset;
                var top = box.Top - labelBoxHeight;
                var right = box.Left + labelWidth + (margin * 2);
                var bottom = box.Top - labelOffset;

                var labelBackground = new SKRectI(left, top, right, bottom);

                // Calculate label text coordinates
                var text_x = labelBackground.Left + margin;
                var text_y = labelBackground.Top + textOffset;

                // Bounding-box
                canvas.DrawRect(box, boxPaint);

                // Label background
                canvas.DrawRect(labelBackground, labelBgPaint);

                // Text shadow
                canvas.DrawText(labelText, text_x + shadowOffset, text_y + shadowOffset, font, ImageConfig.TextShadowPaint);

                // Label text
                canvas.DrawText(labelText, text_x, text_y, font, ImageConfig.FontColorPaint);

                // Draw tail if tracking is enabled
                DrawTrackedTail(canvas, detection.Tail, borderThickness);
            }
        }

        private static void DrawTrackedTail(SKCanvas canvas, List<SKPoint>? tail, float tailThickness)
        {
            // Gradient from solid color to transparent
            var startColor = SKColors.HotPink;
            var endColor = startColor.WithAlpha(0); // Transparent red

            // Bounding box paint
            using var tailPaint = new SKPaint()
            {
                Style = SKPaintStyle.Stroke,
                Color = SKColors.HotPink,
                StrokeWidth = tailThickness * 1.2f,
                IsAntialias = true,
                StrokeCap = SKStrokeCap.Round,
                StrokeJoin = SKStrokeJoin.Round
            };

            var tailLength = tail?.Count();
            if (tail is not null && tailLength is not null && tailLength > 2)
            {
                // Start a new path
                using var path = new SKPath();

                using var shader = SKShader.CreateLinearGradient(
                    tail[^1],
                    tail[0],
                    new[] { startColor, endColor },
                    new float[] { 0, 1 },
                    SKShaderTileMode.Clamp
                );

                // Move "pen" to first position
                path.MoveTo(tail[0]);

                // Move pen along the path with smooth corners
                for (int i = 1; i < tailLength - 2; i++)
                {
                    var midX = (tail[i].X + tail[i + 1].X) / 2;
                    var midY = (tail[i].Y + tail[i + 1].Y) / 2;
                    path.QuadTo(tail[i].X, tail[i].Y, midX, midY);
                }
                path.QuadTo(tail[^2].X, tail[^2].Y, tail[^1].X, tail[^1].Y);

                // Draw path with faded ends on canvas
                tailPaint.Shader = shader;
                canvas.DrawPath(path, tailPaint);
            }
        }

        private static void DrawOrientedBoundingBoxes(this SKBitmap image,
            IEnumerable<OBBDetection>? detections,
            bool drawConfidence,
            SKTypeface fontType = default!)
        {
            ArgumentNullException.ThrowIfNull(detections);

            var fontSize = image.CalculateDynamicSize(ImageConfig.FONT_SIZE);
            var borderThickness = image.CalculateDynamicSize(ImageConfig.BORDER_THICKNESS);
            var margin = (int)ImageConfig.FONT_SIZE / 2;
            var labelBoxHeight = (int)ImageConfig.FONT_SIZE * 2;
            var textOffset = (int)(ImageConfig.FONT_SIZE + margin) - (margin / 2);
            var shadowOffset = ImageConfig.SHADOW_OFFSET;
            byte textShadowAlpha = ImageConfig.DEFAULT_OPACITY;
            byte labelBoxAlpha = ImageConfig.DEFAULT_OPACITY;

            using var font = new SKFont
            {
                Size = fontSize,
                Typeface = fontType ?? SKTypeface.Default
            };

            // Paint buckets
            using var paintText = new SKPaint { IsAntialias = true };
            using var boxPaint = new SKPaint() { Style = SKPaintStyle.Stroke, StrokeWidth = borderThickness };

            using var canvas = new SKCanvas(image);

            foreach (var detection in detections)
            {
                var box = detection.BoundingBox;
                var radians = detection.OrientationAngle;

                var boxColor = HexToRgbaSkia(detection.Label.Color, labelBoxAlpha);
                var labelText = LabelText(detection.Label.Name, detection.Confidence, drawConfidence);

                var labelWidth = (int)font.MeasureText(labelText);

                // Set matrix center point in current bounding box
                canvas.Translate(box.MidX, box.MidY);

                // Rotate image x degrees around the center point
                canvas.RotateRadians(radians);

                // Rotate back
                canvas.Translate(-box.MidX, -box.MidY);

                // Apply rotation and draw bounding box
                boxPaint.Color = boxColor;
                boxPaint.Style = SKPaintStyle.Stroke;
                boxPaint.StrokeWidth = borderThickness;
                canvas.DrawRect(box, boxPaint);

                // Reset matrix, no rotation from this point...
                canvas.SetMatrix(SKMatrix.Identity);

                // Get right bottom corner coordinates after rotation
                var rotationMatrix = SKMatrix.CreateRotation(radians, box.MidX, box.MidY);
                var position = rotationMatrix.MapPoint(new SKPoint(box.Right, box.Bottom));

                // Draw label background
                boxPaint.Style = SKPaintStyle.Fill;
                boxPaint.StrokeWidth = 0;
                canvas.DrawRect(position.X, position.Y, (margin * 2) + labelWidth, labelBoxHeight, boxPaint);

                // Text shadow
                paintText.Color = new SKColor(0, 0, 0, textShadowAlpha);
                canvas.DrawText(labelText, margin + position.X + shadowOffset, textOffset + position.Y + shadowOffset, font, paintText);

                // Label text
                paintText.Color = SKColors.White;
                canvas.DrawText(labelText, margin + position.X, textOffset + position.Y, font, paintText);

            }
        }

        /// <summary>
        /// Converts a hexadecimal color representation to an Rgba32 color.
        /// </summary>
        /// <param name="hexColor">The hexadecimal color value (e.g., "#RRGGBB").</param>
        /// <param name="alpha">Optional. The alpha (transparency) value for the Rgba32 color (0-255, default is 255).</param>
        /// <returns>An Rgba32 color representation.</returns>
        /// <exception cref="ArgumentException">Thrown when the input hex color format is invalid.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when the alpha value is outside the valid range (0-255).</exception>
        private static SKColor HexToRgbaSkia(string hexColor, int alpha = 255)
        {
            var hexValid = SKColor.TryParse(hexColor, out _);

            if (hexColor.Length != 7 || hexValid is false)
                throw new ArgumentException("Invalid hexadecimal color format.");

            if (alpha < 0 || alpha > 255)
                throw new ArgumentOutOfRangeException(nameof(alpha), "Alfa value must be between 0-255.");

            byte r = byte.Parse(hexColor.Substring(1, 2), NumberStyles.HexNumber);
            byte g = byte.Parse(hexColor.Substring(3, 2), NumberStyles.HexNumber);
            byte b = byte.Parse(hexColor.Substring(5, 2), NumberStyles.HexNumber);

            return new SKColor(r, g, b, (byte)alpha);
        }

        #endregion
    }
}
