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
        public static SKImage Draw(this SKImage image, IEnumerable<Classification>? classifications, bool drawConfidence = true)
            => image.DrawClassificationLabels(classifications, drawConfidence);

        /// <summary>
        /// Draw bounding boxes around detected objects on the specified image.
        /// </summary>
        /// <param name="image">The image on which to draw bounding boxes.</param>
        /// <param name="objectDetections">An enumerable collection of objects representing the detected items.</param>
        /// <param name="drawConfidence">A boolean indicating whether to include confidence percentages in the drawn labels.</param>
        public static SKImage Draw(this SKImage image, IEnumerable<ObjectDetection>? objectDetections, bool drawConfidence = true)
            => image.DrawBoundingBoxes(objectDetections, drawConfidence);


        /// <summary>
        /// Draw oriented bounding boxes around detected objects on the specified image.
        /// </summary>
        /// <param name="image">The image on which to draw oriented bounding boxes.</param>
        /// <param name="detections">An enumerable collection of objects representing the detected items.</param>
        /// <param name="drawConfidence">A boolean indicating whether to include confidence percentages in the drawn labels.</param>
        public static SKImage Draw(this SKImage image, IEnumerable<OBBDetection>? detections, bool drawConfidence = true)
            => image.DrawOrientedBoundingBoxes(detections, drawConfidence);

        /// <summary>
        /// Draw segmentations and bounding boxes on the specified image.
        /// </summary>
        /// <param name="image">The image on which to draw segmentations.</param>
        /// <param name="segmentations">A list of segmentation information, including rectangles and segmented pixels.</param>
        /// <param name="draw">Specifies the segments to draw, with a default value.</param>
        /// <param name="drawConfidence">A boolean indicating whether to include confidence percentages in the drawn bounding boxes.</param>
        public static SKImage Draw(this SKImage image, IEnumerable<Segmentation>? segmentations, DrawSegment drawSegment = DrawSegment.Default, bool drawConfidence = true)
            => image.DrawSegmentations(segmentations, drawSegment, drawConfidence);

        /// <summary>
        /// Draws pose-estimated keypoints and bounding boxes on the specified image.
        /// </summary>
        /// <param name="image">The image on which to draw pose estimations.</param>
        /// <param name="poseEstimations">A list of pose estimations.</param>
        /// <param name="keyPointOptions">Options for drawing keypoints.</param>
        /// <param name="drawConfidence">A boolean indicating whether to include confidence percentages in the drawn bounding boxes.</param>
        public static SKImage Draw(this SKImage image, IEnumerable<PoseEstimation>? poseEstimations, KeyPointOptions keyPointOptions, bool drawConfidence = true)
            => image.DrawPoseEstimation(poseEstimations, keyPointOptions, drawConfidence);

        /// <summary>
        /// Saves the SKImage to a file with the specified format and quality.
        /// </summary>
        /// <param name="image">The SKImage to be saved.</param>
        /// <param name="filename">The name of the file where the image will be saved.</param>
        /// <param name="format">The format in which the image should be saved.</param>
        /// <param name="quality">The quality of the saved image (default is 100).</param>
        public static void Save(this SKImage image, string filename, SKEncodedImageFormat format, int quality = 100)
        {
            using var fileStream = new FileStream(
                filename,
                FileMode.OpenOrCreate,
                FileAccess.ReadWrite,
                FileShare.ReadWrite,
                4096,
                true);

            image.Encode(format, quality).SaveTo(fileStream);
        }

        /// <summary>
        /// Creates a resized clone of the input image with new width, height, colorspace (RGB888x) and padded borders to fit the new size.
        /// </summary>
        /// <param name="image">The original image to be resized.</param>
        /// <param name="skInfo">The desired SKImageInfo for the resized image.</param>
        /// <returns>A new image with the specified dimensions.</returns>
        public static SKBitmap ResizeImage(this SKImage image, SKImageInfo skInfo)
        {
            int modelWidth = skInfo.Width;
            int modelHeight = skInfo.Height;
            int width = image.Width;
            int height = image.Height;

            // If the image is already of the correct size and color space, no resizing or conversion is needed
            if (width == modelHeight && height == modelWidth && image.ColorSpace == skInfo.ColorSpace)
                return SKBitmap.FromImage(image);

            // Calculate the new image size based on the aspect ratio
            float scaleFactor = Math.Min((float)modelWidth / width, (float)modelHeight / height);
            int newWidth = (int)Math.Round(width * scaleFactor); // Use integer rounding instead of Math.Round
            int newHeight = (int)Math.Round(height * scaleFactor);

            // Calculate the destination rectangle within the model dimensions
            int x = (modelWidth - newWidth) / 2;
            int y = (modelHeight - newHeight) / 2;

            // Create a new bitmap with the specified SKImageInfo
            var resizedBitmap = new SKBitmap(skInfo);

            // Create a canvas to draw on the new bitmap
            using (var canvas = new SKCanvas(resizedBitmap))
            {
                // Define the source and destination rectangles for resizing
                var srcRect = new SKRect(0, 0, width, height);
                var dstRect = new SKRect(x, y, x + newWidth, y + newHeight);

                // Draw the original image onto the new canvas, resizing it to fit within the destination rectangle
                canvas.DrawImage(image, srcRect, dstRect, new SKPaint { FilterQuality = SKFilterQuality.Low, IsAntialias = false });
                canvas.Flush();
            }

            return resizedBitmap;
        }

        /// <summary>
        /// Extract and normalize pixel values in an image into a DenseTensor object.
        /// </summary>
        /// <param name="img">The image to extract pixel values from.</param>
        /// <returns>A tensor containing normalized pixel values extracted from the input image.</returns>
        public static DenseTensor<float> NormalizePixelsToTensor(this SKBitmap resizedImage, long[] inputShape, int tensorBufferSize, float[] tensorArrayBuffer)
        {
            if (resizedImage.ColorType != SKColorType.Rgb888x)
                throw new ArgumentException("The resized image must be in Rgb888x format.");

            var (batchSize, colorChannels, width, height) = ((int)inputShape[0], (int)inputShape[1], (int)inputShape[2], (int)inputShape[3]);
            var pixelsPerChannel = tensorBufferSize / colorChannels;

            var pixelIndex = 0;
            int offset = 0;

            // Lock the pixels for direct memory access and faster processing
            IntPtr pixelsPtr = resizedImage.GetPixels();

            unsafe
            {
                byte* pixels = (byte*)pixelsPtr;

                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++, pixelIndex++, offset += 4)
                    {
                        offset = (y * width + x) * 4;

                        var r = pixels[offset];
                        var g = pixels[offset + 1];
                        var b = pixels[offset + 2];

                        if ((r | g | b) == 0)
                            continue;

                        tensorArrayBuffer[pixelIndex] = r / 255.0f;
                        tensorArrayBuffer[pixelIndex + pixelsPerChannel] = g / 255.0f;
                        tensorArrayBuffer[pixelIndex + pixelsPerChannel * 2] = b / 255.0f;
                    }
                }
            }

            // Due to how the ArrayPool works, tensorArrayBuffer can be larger than the actual tensor size; we need to cut it down to the correct size.
            return new DenseTensor<float>(tensorArrayBuffer.AsMemory()[..tensorBufferSize], [batchSize, colorChannels, width, height]);
        }

        #region Helper methods

        /// <summary>
        /// Helper method for drawing classification labels.
        /// </summary>
        /// <param name="image">The image on which the labels are to be drawn.</param>
        /// <param name="labels">An collection of classification labels and confidence scores.</param>
        /// <param name="drawConfidence">A flag indicating whether to include confidence scores in the labels.</param>
        public static SKImage DrawClassificationLabels(this SKImage image, IEnumerable<Classification>? labels, bool drawConfidence = true)
        {
            ArgumentNullException.ThrowIfNull(labels);

            float x = ImageConfig.CLASSIFICATION_TRANSPARENT_BOX_X;
            float y = ImageConfig.CLASSIFICATION_TRANSPARENT_BOX_Y;
            var (fontSize, _) = image.CalculateFontSize();
            float margin = fontSize / 2;

            using var paint = new SKPaint
            {
                TextSize = fontSize,
                Style = SKPaintStyle.Fill,
                IsAntialias = true
            };

            // Measure maximum text-length in order to determine the width of the transparent box
            float boxMaxWidth = 0;
            float boxMaxHeight = 0 - margin / 2;
            foreach (var label in labels)
            {
                var lineWidth = paint.MeasureText(LabelText(label.Label, label.Confidence, drawConfidence));
                if (lineWidth > boxMaxWidth)
                    boxMaxWidth = lineWidth;

               boxMaxHeight += fontSize + margin;
            }

            using var surface = SKSurface.Create(new SKImageInfo(image.Width, image.Height));
            using var canvas = surface.Canvas;
            canvas.DrawImage(image, 0, 0);

            // Set transparent box position
            paint.Color = new SKColor(0, 0, 0, ImageConfig.CLASSIFICATION_BOX_ALPHA);
            canvas.DrawRect(new SKRect(x, y, x + boxMaxWidth + fontSize, y + boxMaxHeight + fontSize), paint);

            // Draw labels
            y += paint.TextSize;
            paint.Color = SKColors.White;
            foreach (var label in labels!)
            {
                canvas.DrawText(LabelText(label.Label, label.Confidence, drawConfidence), x + margin, y + margin, paint);
                y += fontSize + margin;
            }
            
            // Finalize drawing
            canvas.Flush();
            return surface.Snapshot();
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
        private static SKImage DrawSegmentations(this SKImage image, IEnumerable<Segmentation>? segmentations, DrawSegment draw, bool drawConfidence)
        {
            ArgumentNullException.ThrowIfNull(segmentations);

            // Convert SKImage to SKBitmap to ensure pixel data is accessible
            SKBitmap bitmap = SKBitmap.FromImage(image);
            IntPtr pixelsPtr = bitmap.GetPixels();
            int width = bitmap.Width;
            int height = bitmap.Height;
            int bytesPerPixel = bitmap.BytesPerPixel;
            int rowBytes = bitmap.RowBytes;

            if (draw == DrawSegment.Default || draw == DrawSegment.PixelMaskOnly)
            {
                var options = new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount };

                Parallel.ForEach(segmentations, options, segmentation =>
                {
                    // Define the overlay color
                    var color = HexToRgbaSkia(segmentation.Label.Color, 80);

                    var pixelSpan = segmentation.SegmentedPixels.AsSpan();

                    unsafe
                    {
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
                    }
                });
            }

            return draw switch
            {
                DrawSegment.PixelMaskOnly => SKImage.FromBitmap(bitmap),
                DrawSegment.BoundingBoxOnly => SKImage.FromBitmap(bitmap).DrawBoundingBoxes(segmentations, drawConfidence),
                _ => SKImage.FromBitmap(bitmap).DrawBoundingBoxes(segmentations, drawConfidence)
            };
        }

        /// <summary>
        /// Helper method for drawing pose estimation and bounding boxes.
        /// </summary>
        /// <param name="image">The image on which to pose estimations.</param>
        /// <param name="poseEstimations">A list of pose estimation information, including rectangles and pose markers.</param>
        /// <param name="drawConfidence">A boolean indicating whether to include confidence percentages in the drawn bounding boxes.</param>
        private static SKImage DrawPoseEstimation(this SKImage image, IEnumerable<PoseEstimation>? poseEstimations, KeyPointOptions poseOptions, bool drawConfidence)
        {
            ArgumentNullException.ThrowIfNull(poseEstimations);

            var options = new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount };

            var circleRadius = ImageConfig.FONT_SIZE_8 / 2;
            var lineSize = 2;
            var confidenceThreshold = poseOptions.PoseConfidence;
            var hasPoseMarkers = poseOptions.PoseMarkers.Length > 0;
            var emptyPoseMarker = new KeyPointMarker();
            var alpha = ImageConfig.POSE_ESTIMATION_MARKER_OPACITY;

            using var keyPointPaint = new SKPaint { Style = SKPaintStyle.Fill, IsAntialias = true };

            using var surface = SKSurface.Create(new SKImageInfo(image.Width, image.Height));
            using var canvas = surface.Canvas;

            canvas.DrawImage(image, 0, 0);

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
                    keyPointPaint.Color = color;
                    canvas.DrawCircle(keyPoint.X, keyPoint.Y, circleRadius, keyPointPaint);

                    // Draw lines between pose-markers
                    foreach (var connection in poseMap.Connections)
                    {
                        var markerDestination = poseEstimation.KeyPoints[connection.Index];

                        if (markerDestination.Confidence < confidenceThreshold)
                            continue;

                        //var destination = new Point(markerDestination.X, markerDestination.Y);
                        var lineColor = HexToRgbaSkia(connection.Color);

                        var fromKeyPoint = new SKPoint(keyPoint.X, keyPoint.Y);
                        var toKeyPoint = new SKPoint(markerDestination.X, markerDestination.Y);
                        canvas.DrawLine(fromKeyPoint, toKeyPoint, new SKPaint { Color = lineColor, StrokeWidth = lineSize, IsAntialias = true });
                    }
                }
            }

            canvas.Flush();

            if (poseOptions.DrawBoundingBox)
                return surface.Snapshot().DrawBoundingBoxes(poseEstimations, drawConfidence);

            return surface.Snapshot();
        }

        /// <summary>
        /// Helper method for drawing bounding boxes around detected objects.
        /// </summary>
        /// <param name="image">The image on which to draw bounding boxes.</param>
        /// <param name="detections">An enumerable collection of objects representing the detected items.</param>
        /// <param name="drawConfidence">A boolean indicating whether to include confidence percentages in the drawn labels.</param>
        private static SKImage DrawBoundingBoxes(this SKImage image, IEnumerable<IDetection>? detections, bool drawConfidence)
        {
            ArgumentNullException.ThrowIfNull(detections);

            var (fontSize, borderThickness) = image.CalculateFontSize();

            //float fontSize = image.CalculateFontSize(ImageConfig.DEFAULT_FONT_SIZE);
            var margin = (int)fontSize / 2;
            var labelBoxHeight = (int)fontSize * 2;
            var textOffset = (int)(fontSize + margin) - (margin / 2);
            var shadowOffset = ImageConfig.SHADOW_OFFSET;
            var labelOffset = (int)borderThickness / 2;
            byte textShadowAlpha = ImageConfig.DEFAULT_OPACITY;
            byte labelBoxAlpha = ImageConfig.DEFAULT_OPACITY;

            // Shadow paint
            using var paintShadow = new SKPaint
            {
                TextSize = fontSize, //ImageConfig.DEFAULT_FONT_SIZE,
                Color = new SKColor(0, 0, 0, textShadowAlpha),
                IsAntialias = true
            };

            // Text paint
            using var paintText = new SKPaint
            {
                TextSize = fontSize, //ImageConfig.DEFAULT_FONT_SIZE,
                Color = SKColors.White,
                IsAntialias = true
            };

            // Label box background paint
            using var labelBgPaint = new SKPaint
            {
                Style = SKPaintStyle.Fill,
                StrokeWidth = borderThickness
            };

            // Bounding box paint
            using var boxPaint = new SKPaint()
            {
                Style = SKPaintStyle.Stroke,
                StrokeWidth = borderThickness
            };

            using var surface = SKSurface.Create(new SKImageInfo(image.Width, image.Height, SKColorType.Rgba8888));
            using var canvas = surface.Canvas;

            // Draw image on surface
            canvas.DrawImage(image, 0, 0);

            // Draw detections
            foreach (var detection in detections)
            {
                var box = detection.BoundingBox;
                var boxColor = HexToRgbaSkia(detection.Label.Color, labelBoxAlpha);
                var labelText = LabelText(detection.Label.Name, detection.Confidence, drawConfidence);
                var labelWidth = (int)paintText.MeasureText(labelText);

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
                canvas.DrawText(labelText, text_x + shadowOffset, text_y + shadowOffset, paintShadow);

                // Label text
                canvas.DrawText(labelText, text_x, text_y, paintText);
            }

            // Execute all pending draw operations
            canvas.Flush();

            return surface.Snapshot();
        }

        private static SKImage DrawOrientedBoundingBoxes(this SKImage image, IEnumerable<OBBDetection>? detections, bool drawConfidence)
        {
            ArgumentNullException.ThrowIfNull(detections);

            var (fontSize, borderThickness) = image.CalculateFontSize();
            var margin = (int)ImageConfig.FONT_SIZE / 2;
            var labelBoxHeight = (int)ImageConfig.FONT_SIZE * 2;
            var textOffset = (int)(ImageConfig.FONT_SIZE + margin) - (margin / 2);
            var shadowOffset = ImageConfig.SHADOW_OFFSET;
            byte textShadowAlpha = ImageConfig.DEFAULT_OPACITY;
            byte labelBoxAlpha = ImageConfig.DEFAULT_OPACITY;

            // Paint buckets
            using var paintText = new SKPaint { TextSize = fontSize, IsAntialias = true };
            using var boxPaint = new SKPaint() { Style = SKPaintStyle.Stroke, StrokeWidth = borderThickness };

            // Create surface
            using var surface = SKSurface.Create(new SKImageInfo(image.Width, image.Height, SKColorType.Rgba8888));
            using var canvas = surface.Canvas;

            // Draw image on surface
            canvas.DrawImage(image, 0, 0);

            foreach (var detection in detections)
            {
                var box = detection.BoundingBox;
                var radians = detection.OrientationAngle;

                var boxColor = HexToRgbaSkia(detection.Label.Color, labelBoxAlpha);
                var labelText = LabelText(detection.Label.Name, detection.Confidence, drawConfidence);
                var labelWidth = (int)paintText.MeasureText(labelText);

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
                canvas.DrawText(labelText, margin + position.X + shadowOffset, textOffset + position.Y + shadowOffset, paintText);

                // Label text
                paintText.Color = SKColors.White;
                canvas.DrawText(labelText, margin + position.X, textOffset + position.Y, paintText);
            }

            canvas.Flush();
            return surface.Snapshot();
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
