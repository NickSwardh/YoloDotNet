// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Extensions
{
    public static class ImageExtension
    {
        /// <summary>
        /// Draws classification labels on the given <see cref="SKBitmap"/>.
        /// This method modifies the bitmap in place.
        /// </summary>
        /// <param name="image">The image on which to draw labels.</param>
        /// <param name="classifications">A collection of classification results.</param>
        /// <param name="options">Drawing options that control font family, size, color.</param>
        public static void Draw(this SKBitmap image, IEnumerable<Classification>? classifications, ClassificationDrawingOptions options = default!)
            => image.DrawClassificationLabels(classifications, options);

        /// <summary>
        /// Draws classification labels on the given <see cref="SKImage"/>.
        /// </summary>
        /// <param name="image">The image from which to create a bitmap before drawing the labels.</param>
        /// <param name="classifications">A collection of classification results containing labels and confidence scores.</param>
        /// <param name="options">Drawing options that control font family, size, color.</param>
        /// <returns>A new <see cref="SKBitmap"/> with classification labels drawn on it; the original image remains unmodified.</returns>
        public static SKBitmap Draw(this SKImage image, IEnumerable<Classification>? classifications, ClassificationDrawingOptions options = default!)
        {
            var img = SKBitmap.FromImage(image);
            img.DrawClassificationLabels(classifications, options);

            return img;
        }

        /// <summary>
        /// Draws bounding boxes around detected objects on the given <see cref="SKBitmap"/>.
        /// This method modifies the bitmap in place.
        /// </summary>
        /// <param name="image">The image on which to draw bounding boxes.</param>
        /// <param name="objectDetections">An enumerable collection of objects representing the detected items.</param>
        /// <param name="options">Drawing options that control bounding box appearance, labels, confidence scores, fonts, and other visuals.</param>
        public static void Draw(this SKBitmap image, IEnumerable<ObjectDetection>? objectDetections, DetectionDrawingOptions options = default!)
            => image.DrawBoundingBoxes(objectDetections, options);

        /// <summary>
        /// Draws bounding boxes around detected objects on the given <see cref="SKImage"/>.
        /// </summary>
        /// <param name="image">The image on which to draw bounding boxes.</param>
        /// <param name="objectDetections">An enumerable collection of objects representing the detected items.</param>
        /// <param name="options">Drawing options that control bounding box appearance, labels, confidence scores, fonts, and other visuals.</param>
        /// <returns>A new <see cref="SKBitmap"/> with bounding boxes drawn on it.</returns>
        public static SKBitmap Draw(this SKImage image, IEnumerable<ObjectDetection>? objectDetections, DetectionDrawingOptions options = default!)
        {
            var img = SKBitmap.FromImage(image);
            img.DrawBoundingBoxes(objectDetections, options);

            return img;
        }

        /// <summary>
        /// Draws oriented bounding boxes (OBBs) around detected objects on the given <see cref="SKBitmap"/>.
        /// This method modifies the bitmap in place.
        /// </summary>
        /// <param name="image">The image on which to draw oriented bounding boxes.</param>
        /// <param name="detections">An enumerable collection of objects representing the detected items.</param>
        /// <param name="options">Drawing options that control bounding box appearance, labels, confidence scores, fonts, and other visuals.</param>
        public static void Draw(this SKBitmap image, IEnumerable<OBBDetection>? detections, DetectionDrawingOptions options = default!)
            => image.DrawOrientedBoundingBoxes(detections, options);

        /// <summary>
        /// Draws oriented bounding boxes (OBBs) around detected objects on the given <see cref="SKImage"/>.
        /// </summary>
        /// <param name="image">The image on which to draw oriented bounding boxes.</param>
        /// <param name="detections">An enumerable collection of objects representing the detected items.</param>
        /// <param name="options">Drawing options that control bounding box appearance, labels, confidence scores, fonts, and other visuals.</param>
        /// <returns>
        /// A new <see cref="SKBitmap"/> with oriented bounding boxes and optional confidence labels drawn on it.
        /// </returns>
        public static SKBitmap Draw(this SKImage image, IEnumerable<OBBDetection>? detections, DetectionDrawingOptions options = default!)
        {
            var img = SKBitmap.FromImage(image);
            img.DrawOrientedBoundingBoxes(detections, options);

            return img;
        }

        /// <summary>
        /// Draws segmentation masks and bounding boxes on the specified <see cref="SKBitmap"/>.
        /// This method modifies the bitmap in place by overlaying the selected segments and labels.
        /// </summary>
        /// <param name="image">The image on which to draw segmentations.</param>
        /// <param name="segmentations">A list of segmentation information, including rectangles and segmented pixels.</param>
        /// <param name="options">Drawing options that control segmentation mas, bounding box appearance, labels, confidence scores, fonts, and other visuals.</param>
        public static void Draw(this SKBitmap image, IEnumerable<Segmentation>? segmentations, SegmentationDrawingOptions options = default!)
            => image.DrawSegmentations(segmentations, options);

        /// <summary>
        /// Draws segmentation masks and bounding boxes on the specified <see cref="SKImage"/>.
        /// </summary>
        /// <param name="image">The image on which to draw segmentations.</param>
        /// <param name="segmentations">A list of segmentation information, including rectangles and segmented pixels.</param>
        /// <param name="options">Drawing options that control segmentation mas, bounding box appearance, labels, confidence scores, fonts, and other visuals.</param>
        /// <returns>
        /// A new <see cref="SKBitmap"/> with the selected segmentation overlays and optional confidence labels drawn.
        /// </returns>
        public static SKBitmap Draw(this SKImage image, IEnumerable<Segmentation>? segmentations, SegmentationDrawingOptions options = default!)
        {
            var img = SKBitmap.FromImage(image);
            img.DrawSegmentations(segmentations, options);

            return img;
        }

        /// <summary>
        /// Draws pose-estimated keypoints and bounding boxes directly on the provided <see cref="SKBitmap"/>.
        /// This method mutates the bitmap by overlaying joint points, skeleton connections, and optional bounding boxes.
        /// </summary>
        /// <param name="image">The image on which to draw pose estimations.</param>
        /// <param name="poseEstimations">A list of pose estimations.</param>
        /// <param name="options">Drawing options that control keypoints, bounding box appearance, labels, confidence scores, fonts, and other visuals.</param>
        public static void Draw(this SKBitmap image, IEnumerable<PoseEstimation>? poseEstimations, PoseDrawingOptions options = default!)
            => image.DrawPoseEstimation(poseEstimations, options);

        /// <summary>
        /// Draws pose-estimated keypoints and bounding boxes on a copy of the given <see cref="SKImage"/>.
        /// This method creates and returns a new <see cref="SKBitmap"/> with the full pose visualization applied.
        /// </summary>
        /// <param name="image">The image on which to draw pose estimations.</param>
        /// <param name="poseEstimations">A list of pose estimations.</param>
        /// <param name="options">Drawing options that control keypoints, bounding box appearance, labels, confidence scores, fonts, and other visuals.</param>
        /// <returns>
        /// A new <see cref="SKBitmap"/> containing the pose estimation drawings without modifying the original <see cref="SKImage"/>.
        /// </returns>
        public static SKBitmap Draw(this SKImage image, IEnumerable<PoseEstimation>? poseEstimations, PoseDrawingOptions options = default!)
        {
            var img = SKBitmap.FromImage(image);
            img.DrawPoseEstimation(poseEstimations, options);

            return img;
        }

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

        /// <summary>
        /// Saves the SKImage to a file with the specified format and quality.
        /// </summary>
        /// <param name="image">The SKImage to be saved.</param>
        /// <param name="filename">The name of the file where the image will be saved.</param>
        /// <param name="format">The format in which the image should be saved.</param>
        /// <param name="quality">The quality of the saved image (default is 100).</param>
        public static void Save(this SKImage image,
            string filename,
            SKEncodedImageFormat format = SKEncodedImageFormat.Jpeg,
            int quality = 100)
            => FrameSaveService.AddToQueue(image, filename, format, quality);

        #region Helper methods

        /// <summary>
        /// Helper method for drawing classification labels.
        /// </summary>
        /// <param name="image">The image on which the labels are to be drawn.</param>
        /// <param name="labels">An collection of classification labels and confidence scores.</param>
        /// <param name="options">Drawing options to control appearance.</param>
        private static void DrawClassificationLabels(this SKBitmap image, IEnumerable<Classification>? labels, ClassificationDrawingOptions options)
        {
            ArgumentNullException.ThrowIfNull(labels);

            options ??= ImageConfig.DefaultClassificationDrawingOptions;

            var drawConfidence = true;

            using var font = new SKFont
            {
                Typeface = options.Font,
                Size = options.FontSize,
            };

            using var fontColor = new SKPaint
            {
                Color = options.FontColor,
                IsAntialias = true
            };

            var fontSize = options?.EnableDynamicScaling ?? true
                ? image.CalculateDynamicSize(font.Size)
                : font.Size;

            float x = ImageConfig.CLASSIFICATION_TRANSPARENT_BOX_X;
            float y = ImageConfig.CLASSIFICATION_TRANSPARENT_BOX_Y;

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
            if (options!.DrawLabelBackground)
                canvas.DrawRect(SKRect.Create(x, y, boxMaxWidth + fontSize, boxMaxHeight + fontSize), ImageConfig.ClassificationBackgroundPaint);

            // Draw labels on transparent box
            y += font.Size;
            foreach (var label in labels!)
            {
                var text = LabelText(label.Label, label.Confidence, drawConfidence);

                // Text shadow
                if (options!.EnableFontShadow)
                    canvas.DrawText(text, x + margin + ImageConfig.SHADOW_OFFSET, y + margin + ImageConfig.SHADOW_OFFSET, font, ImageConfig.TextShadowPaint);

                canvas.DrawText(text, x + margin, y + margin, font, fontColor);
                y += fontSize + margin;
            }
        }

        private static string LabelText(string labelName, double confidence, bool showConfidence)
        {
            var confidenceFormat = showConfidence ? $" ({confidence.ToPercent()}%)" : "";
            return $"{labelName}{confidenceFormat}";
        }

        unsafe private static void DrawSegmentations(this SKBitmap image, IEnumerable<Segmentation>? segmentations, SegmentationDrawingOptions options)
        {
            ArgumentNullException.ThrowIfNull(segmentations);

            options ??= ImageConfig.DefaultSegmentationDrawingOptions;

            // Apply pixelmap on original image
            using var canvas = new SKCanvas(image);

            if (options.DrawSegmentationPixelMask is true)
            {
                var totalColors = options.BoundingBoxHexColors.Length;

                foreach (var segmentation in segmentations)
                {
                    var box = segmentation.BoundingBox;

                    var pixelMask = segmentation.BitPackedPixelMask.UnpackToBitmap(box.Width, box.Height);

                    // Get class color
                    var hexColor = options.BoundingBoxHexColors[segmentation.Label.Index % totalColors];
                    var color = HexToRgbaSkia(hexColor, options.BoundingBoxOpacity);

                    using var paint = new SKPaint
                    {
                        ColorFilter = CreateGrayscaleToColorFilter(color),
                        BlendMode = SKBlendMode.SrcOver // Black pixels will be transparent
                    };

                    // Draw the unpacked mask over original
                    canvas.DrawBitmap(pixelMask, box.Left, box.Top, paint);
                }
            }

            if (options.DrawBoundingBoxes is true)
                image.DrawBoundingBoxes(segmentations, (DetectionDrawingOptions)options);
        }

        /// <summary>
        /// Helper method for drawing pose estimation and bounding boxes.
        /// </summary>
        /// <param name="image">The image on which to draw pose estimation results.</param>
        /// <param name="poseEstimations">A list of pose estimation information, including rectangles and pose markers.</param>
        /// <param name="options">Drawing options to control appearance.</param>
        private static void DrawPoseEstimation(this SKBitmap image, IEnumerable<PoseEstimation>? poseEstimations, PoseDrawingOptions options)
        {
            ArgumentNullException.ThrowIfNull(poseEstimations);

            options ??= ImageConfig.DefaultPoseDrawingOptions;

            // If no keypoints are defined and only bounding boxes should be drawn, render bounding boxes and return early.
            if (options.KeyPointMarkers.Length == 0 && options.DrawBoundingBoxes)
            {
                image.DrawBoundingBoxes(poseEstimations, (DetectionDrawingOptions)options);
                return;
            }

            var circleRadius = image.CalculateDynamicSize(ImageConfig.KEYPOINT_SIZE);
            var lineSize = image.CalculateDynamicSize(options!.BorderThickness);
            var confidenceThreshold = options.PoseConfidence;
            var hasPoseMarkers = options.KeyPointMarkers.Length > 0;
            var emptyPoseMarker = new KeyPointMarker();
            var alpha = options!.BoundingBoxOpacity;

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
                        ? options.KeyPointMarkers[i]
                        : emptyPoseMarker;

                    var color = hasPoseMarkers
                        ? HexToRgbaSkia(poseMap.Color, alpha)
                        : options.DefaultPoseColor;

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

            if (options.DrawBoundingBoxes)
                image.DrawBoundingBoxes(poseEstimations, (DetectionDrawingOptions)options);
        }

        /// <summary>
        /// Helper method for drawing bounding boxes around detected objects.
        /// </summary>
        /// <param name="image">The image on which to draw bounding boxes.</param>
        /// <param name="detections">An enumerable collection of objects representing the detected items.</param>
        /// <param name="options">Drawing options to control appearance.</param>
        private static void DrawBoundingBoxes(this SKBitmap image, IEnumerable<IDetection>? detections, DetectionDrawingOptions options)
        {
            ArgumentNullException.ThrowIfNull(detections);

            // Custom or default options for drawing?
            options ??= ImageConfig.DefaultDetectionDrawingOptions;

            using var font = new SKFont
            {
                Typeface = options.Font,
                Size = options.FontSize,
            };

            using var fontColor = new SKPaint
            {
                Color = options.FontColor,
                IsAntialias = true
            };

            var fontSize = options?.EnableDynamicScaling ?? true
                ? image.CalculateDynamicSize(font.Size)
                : font.Size;

            var borderThickness = options!.BorderThickness;

            // Should font and border size be sized dynamically based on image dimension?
            if (options.EnableDynamicScaling is true)
            {
                fontSize = image.CalculateDynamicSize(font.Size);
                borderThickness = image.CalculateDynamicSize(options!.BorderThickness);

                // Update font size
                font.Size = fontSize;
            }

            var margin = (int)fontSize / 2;
            var labelBoxHeight = (int)fontSize * 2;
            var textOffset = (int)(fontSize + margin) - (margin / 2);
            var shadowOffset = ImageConfig.SHADOW_OFFSET;
            var labelOffset = (int)borderThickness / 2;

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

            var totalColors = options.BoundingBoxHexColors.Length;

            // Draw detections
            foreach (var detection in detections)
            {
                var box = detection.BoundingBox;

                var hex = options.BoundingBoxHexColors[detection.Label.Index % totalColors];
                var boxColor = HexToRgbaSkia(hex, options.BoundingBoxOpacity);

                var text = (detection.Id is not null)
                    ? $"Id: {detection.Id}, {detection.Label.Name}"
                    : detection.Label.Name;

                var labelText = LabelText(text, detection.Confidence, options.DrawConfidenceScore);
                var labelWidth = (int)font.MeasureText(labelText);

                labelBgPaint.Color = boxColor;
                boxPaint.Color = boxColor;

                // Calculate label background rect size
                var left = box.Left - labelOffset;
                var top = box.Top - labelBoxHeight;
                var right = box.Left + labelWidth + (margin * 2);
                var bottom = box.Top - labelOffset;

                var labelBackground = new SKRectI(left, top, right, bottom);

                // Bounding-box
                if (options.DrawBoundingBoxes)
                    canvas.DrawRect(box, boxPaint);

                // Label text
                if (options.DrawLabels)
                {
                    // Calculate label text coordinates
                    var text_x = labelBackground.Left;// + margin;
                    var text_y = labelBackground.Top + textOffset;

                    // Label background
                    if (options.DrawLabelBackground)
                    {
                        text_x += margin;
                        //text_y += textOffset;
                        canvas.DrawRect(labelBackground, labelBgPaint);
                    }

                    // Text shadow
                    if (options.EnableFontShadow)
                        canvas.DrawText(labelText, text_x + shadowOffset, text_y + shadowOffset, font, ImageConfig.TextShadowPaint);

                    canvas.DrawText(labelText, text_x, text_y, font, fontColor);
                }

                // Draw tail if tracking is enabled
                if (options.DrawTrackedTail is true)
                {
                    DrawTrackedTail(canvas, detection.Tail, options);
                }
            }
        }

        /// <summary>
        /// Helper method for drawing a tail on tracked objects.
        /// </summary>
        /// <param name="image">The image on which to draw the tracked tail.</param>
        /// <param name="detections">A collection of tail-points.</param>
        /// <param name="options">Drawing options to control appearance.</param>
        private static void DrawTrackedTail(SKCanvas canvas, List<SKPoint>? tail, DetectionDrawingOptions options)
        {
            // Bounding box paint
            using var tailPaint = new SKPaint()
            {
                Style = SKPaintStyle.Stroke,
                StrokeWidth = options.TailThickness,
                IsAntialias = true,
                StrokeCap = SKStrokeCap.Round,
                StrokeJoin = SKStrokeJoin.Round
            };

            var tailLength = tail?.Count;
            if (tail is not null && tailLength is not null && tailLength > 2)
            {
                // Start a new path
                using var path = new SKPath();

                using var shader = SKShader.CreateLinearGradient(
                    tail[^1],
                    tail[0],
                    new[] { ImageConfig.TailPaintColorStart, ImageConfig.TailPaintColorEnd }, // Gradient tail color
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

        /// <summary>
        /// Helper method for drawing oriented bounding boxes around detected objects.
        /// </summary>
        /// <param name="image">The image on which to draw the oriented bounding boxes.</param>
        /// <param name="detections">A collection of oriented bounding box detections.</param>
        /// <param name="options">Drawing options to control appearance.</param>
        private static void DrawOrientedBoundingBoxes(this SKBitmap image, IEnumerable<OBBDetection>? detections, DetectionDrawingOptions options)
        {
            ArgumentNullException.ThrowIfNull(detections);

            // Custom or default options for drawing?
            options ??= ImageConfig.DefaultDetectionDrawingOptions;

            using var font = new SKFont
            {
                Typeface = options.Font,
                Size = options.FontSize,
            };

            using var fontColor = new SKPaint
            {
                Color = options.FontColor,
                IsAntialias = true
            };

            var fontSize = font.Size;
            var borderThickness = options.BorderThickness;

            // Should font and border size be sized dynamically based on image dimension?
            if (options.EnableDynamicScaling is true)
            {
                fontSize = image.CalculateDynamicSize(font.Size);
                borderThickness = image.CalculateDynamicSize(options.BorderThickness);

                // Update font size
                font.Size = fontSize;
            }

            var margin = (int)font.Size / 2;
            var labelBoxHeight = (int)font.Size * 2;
            var textOffset = (int)(font.Size + margin) - (margin / 2);
            var shadowOffset = ImageConfig.SHADOW_OFFSET;
            int labelBoxAlpha = options.BoundingBoxOpacity;

            // Paint buckets
            using var boxPaint = new SKPaint() { Style = SKPaintStyle.Stroke, StrokeWidth = borderThickness };

            using var canvas = new SKCanvas(image);

            var totalColors = options.BoundingBoxHexColors.Length;

            foreach (var detection in detections)
            {
                var box = detection.BoundingBox;
                var radians = detection.OrientationAngle;

                var hex = options.BoundingBoxHexColors[detection.Label.Index % totalColors];

                // Get hex color
                var boxColor = HexToRgbaSkia(hex, labelBoxAlpha);
                boxPaint.Color = boxColor;
                boxPaint.Style = SKPaintStyle.Stroke;
                boxPaint.StrokeWidth = borderThickness;

                var labelText = LabelText(detection.Label.Name, detection.Confidence, options.DrawConfidenceScore);

                var labelWidth = (int)font.MeasureText(labelText);

                // Draw rotated bounding box
                if (options.DrawBoundingBoxes)
                {
                    // Set matrix center point in current bounding box
                    canvas.Translate(box.MidX, box.MidY);

                    // Rotate image x degrees around the center point
                    canvas.RotateRadians(radians);

                    // Rotate back
                    canvas.Translate(-box.MidX, -box.MidY);

                    // Apply rotation and draw bounding box
                    canvas.DrawRect(box, boxPaint);

                    // Reset matrix, no rotation from this point...
                    canvas.SetMatrix(SKMatrix.Identity);
                }

                // Get right bottom corner coordinates after rotation
                var rotationMatrix = SKMatrix.CreateRotation(radians, box.MidX, box.MidY);
                var position = rotationMatrix.MapPoint(new SKPoint(box.Right, box.Bottom));

                // Label text
                if (options.DrawLabels)
                {
                    // Draw label background
                    if (options.DrawLabelBackground)
                    {
                        boxPaint.Style = SKPaintStyle.Fill;
                        boxPaint.StrokeWidth = 0;
                        canvas.DrawRect(position.X, position.Y, (margin * 2) + labelWidth, labelBoxHeight, boxPaint);
                    }

                    // Draw text shadow
                    if (options.EnableFontShadow)
                    {
                        canvas.DrawText(labelText, margin + position.X + shadowOffset, textOffset + position.Y + shadowOffset, font, ImageConfig.TextShadowPaint);
                    }

                    canvas.DrawText(labelText, margin + position.X, textOffset + position.Y, font, fontColor);
                }
            }
        }

        public static byte[] UnpackPixelMaskToByteArray(this byte[] packedMask, int width, int height)
        {
            int totalPixels = width * height;
            byte[] unpacked = new byte[totalPixels];

            for (int i = 0; i < totalPixels; i++)
            {
                // Use bitwise comparison instead of / (division) and % (modulus) for faster calculations
                int byteIndex = i >> 3;     // i / 8
                int bitIndex = i & 7;       // i % 8

                // Set each unpacked pixel to either black or white
                bool isSet = (packedMask[byteIndex] & (1 << bitIndex)) != 0;
                unpacked[i] = isSet ? (byte)255 : (byte)0;
            }

            return unpacked;
        }

        unsafe public static SKBitmap UnpackToBitmap(this byte[] packedMask, int width, int height)
        {
            // Create a bitmap for the pixelmask
            var bitmap = new SKBitmap(width, height, SKColorType.Bgra8888, SKAlphaType.Premul);

            // Caclulate total pixels
            var totalPixels = width * height;

            // Get direct access to the bitmap pixels for faster drawing
            byte* ptr = (byte*)bitmap.GetPixels().ToPointer();

            // Iterate through all pixels
            for (int i = 0; i < totalPixels; i++)
            {
                // Use bitwise operations for speed (i / 8 and i % 8)
                int byteIndex = i >> 3;      // i / 8
                int bitIndex = i & 0b0111;   // i % 8

                // Mask check
                bool isOn = (packedMask[byteIndex] & (1 << bitIndex)) != 0;

                // This will be 255 (white) if true, 0 (black) if false
                byte color = isOn ? (byte)255 : (byte)0;

                // Calculate the starting offset in the byte array (4 bytes per pixel)
                int offset = i * 4;

                // Set color with full alpha (fully visible)
                ptr[offset + 0] = color; // Blue
                ptr[offset + 1] = color; // Green
                ptr[offset + 2] = color; // Red
                ptr[offset + 3] = color; // Alpha
            }

            return bitmap;
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
                throw new ArgumentOutOfRangeException(nameof(alpha), "Alpha value must be between 0-255.");

            byte r = byte.Parse(hexColor.Substring(1, 2), NumberStyles.HexNumber);
            byte g = byte.Parse(hexColor.Substring(3, 2), NumberStyles.HexNumber);
            byte b = byte.Parse(hexColor.Substring(5, 2), NumberStyles.HexNumber);

            return new SKColor(r, g, b, (byte)alpha);
        }

        public static SKColorFilter CreateGrayscaleToColorFilter(SKColor color)
        {
            float r = color.Red / 255f;
            float g = color.Green / 255f;
            float b = color.Blue / 255f;
            float alpha = color.Alpha / 255f;

            return SKColorFilter.CreateColorMatrix(new[]
            {
                r, 0, 0, 0, 0,
                g, 0, 0, 0, 0,
                b, 0, 0, 0, 0,
                0, 0, 0, alpha, 0
            });
        }

        #endregion
    }
}
