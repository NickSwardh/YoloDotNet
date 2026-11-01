// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Tests.DrawingTests
{
    /// <summary>
    /// The AVX2 fill used by Draw and the pixel masks, checked against exact geometry and against SkiaSharp.
    /// Every test returns early on CPUs without AVX2, where Draw and the masks use SkiaSharp instead.
    /// </summary>
    public class Avx2ShapeFillerTests
    {
        private static readonly SKColor Background = new(100, 150, 200);

        // Edge pixels get the exact covered area, so they may differ from a SkiaSharp drawing of the same shape by a few levels
        private const int EDGE_TOLERANCE = 32;

        // Circles are polygons within about 0.1 px of the true circle, so an edge pixel can be off by about that share of 255
        private const int CIRCLE_TOLERANCE = 20;

        [Fact]
        public void FillPolygons_Rectangle_MatchesExactCoverage()
        {
            if (!Avx2ShapeFiller.IsSupported)
                return;

            using var image = CreateImage(48, 32, SKColors.White);
            float left = 10.2f, top = 5.4f, right = 30.7f, bottom = 20.6f;

            Assert.True(Avx2ShapeFiller.TryFillPolygons(image, [Rectangle(left, top, right, bottom)], SKColors.Black));

            // Black, fully opaque over white: each pixel's darkness is exactly its covered area
            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    float covered = Overlap(x, x + 1, left, right) * Overlap(y, y + 1, top, bottom);
                    int expected = 255 - (int)(covered * 255 + 0.5f);
                    Assert.InRange(image.GetPixel(x, y).Red, expected - 1, expected + 1);
                }
            }
        }

        [Fact]
        public void FillPolygons_Star_MatchesSkia()
        {
            if (!Avx2ShapeFiller.IsSupported)
                return;

            SKPoint[][] star = [Star(new SKPoint(40.3f, 30.6f), 26.2f, 11.7f, 7)];
            var color = new SKColor(230, 40, 90, 128);

            Assert.InRange(MaxDifferenceFromSkia(80, 64, star, color), 0, EDGE_TOLERANCE);
        }

        [Fact]
        public void FillPolygons_ShapeCrossingImageBorders_MatchesSkia()
        {
            if (!Avx2ShapeFiller.IsSupported)
                return;

            // A diamond with all four corners outside the image
            SKPoint[][] diamond = [[new(-15.3f, 20.2f), new(30.4f, -12.7f), new(77.6f, 21.9f), new(31.1f, 58.3f)]];

            Assert.InRange(MaxDifferenceFromSkia(64, 40, diamond, new SKColor(20, 200, 120, 160)), 0, EDGE_TOLERANCE);
        }

        [Fact]
        public void FillPolygons_HoleStaysEmpty()
        {
            if (!Avx2ShapeFiller.IsSupported)
                return;

            using var image = CreateImage(40, 40, Background);

            // Outer and inner outline wound the same way: the even-odd rule still leaves the inside empty
            SKPoint[][] ring = [Rectangle(4, 4, 36, 36), Rectangle(14, 14, 26, 26)];

            Assert.True(Avx2ShapeFiller.TryFillPolygons(image, ring, new SKColor(255, 0, 0, 128)));
            Assert.Equal(Background, image.GetPixel(20, 20));
            Assert.NotEqual(Background, image.GetPixel(8, 20));
        }

        [Theory]
        [InlineData(3f)]
        [InlineData(5f)]
        [InlineData(15.5f)]
        [InlineData(40f)]
        public void FillCircle_MatchesTheExactCircle(float radius)
        {
            if (!Avx2ShapeFiller.IsSupported)
                return;

            int size = (int)(radius * 2) + 12;
            var center = new SKPoint(size / 2f + 0.37f, size / 2f - 0.21f);

            using var image = CreateImage(size, size, SKColors.White);
            Assert.True(Avx2ShapeFiller.TryFillCircle(image, center, radius, SKColors.Black));

            double area = 0;

            for (int y = 0; y < size; y++)
            {
                for (int x = 0; x < size; x++)
                {
                    int darkness = 255 - image.GetPixel(x, y).Red;
                    area += darkness / 255.0;

                    // Each pixel's darkness is how much of it the true circle covers, within the polygon's 0.1 px tolerance
                    int expected = (int)Math.Round(255 * CircleCoverage(x, y, center, radius));
                    Assert.InRange(darkness, expected - CIRCLE_TOLERANCE, expected + CIRCLE_TOLERANCE);
                }
            }

            // The filled area is the circle's area
            Assert.InRange(area / (Math.PI * radius * radius), 0.995, 1.005);
        }

        [Fact]
        public void FillPolygons_ShapeOutsideImage_ChangesNothing()
        {
            if (!Avx2ShapeFiller.IsSupported)
                return;

            using var image = CreateImage(20, 20, Background);

            Assert.True(Avx2ShapeFiller.TryFillPolygons(image, [Rectangle(30, 30, 50, 50)], SKColors.Red));
            Assert.True(Avx2ShapeFiller.TryFillPolygons(image, [Rectangle(-30, 5, -10, 15)], SKColors.Red));
            Assert.True(Avx2ShapeFiller.TryFillPolygons(image, [], SKColors.Red));

            for (int y = 0; y < 20; y++)
                for (int x = 0; x < 20; x++)
                    Assert.Equal(Background, image.GetPixel(x, y));
        }

        [Fact]
        public void FillPolygons_PointThatIsNotANumber_ReturnsFalseAndChangesNothing()
        {
            if (!Avx2ShapeFiller.IsSupported)
                return;

            using var image = CreateImage(20, 20, Background);
            SKPoint[][] broken = [[new(2, 2), new(float.NaN, 5), new(15, 15)]];

            Assert.False(Avx2ShapeFiller.TryFillPolygons(image, broken, SKColors.Red));
            Assert.False(Avx2ShapeFiller.TryFillCircle(image, new SKPoint(10, 10), float.PositiveInfinity, SKColors.Red));

            for (int y = 0; y < 20; y++)
                for (int x = 0; x < 20; x++)
                    Assert.Equal(Background, image.GetPixel(x, y));
        }

        [Theory]
        [InlineData(SKColorType.Gray8)]
        [InlineData(SKColorType.Rgb565)]
        public void FillPolygons_UnsupportedImage_ReturnsFalseAndChangesNothing(SKColorType colorType)
        {
            using var image = new SKBitmap(new SKImageInfo(16, 16, colorType, SKAlphaType.Opaque));
            image.Erase(SKColors.Gray);
            var before = image.GetPixelSpan().ToArray();

            Assert.False(Avx2ShapeFiller.TryFillPolygons(image, [Rectangle(2, 2, 12, 12)], SKColors.Red));
            Assert.False(Avx2ShapeFiller.TryFillCircle(image, new SKPoint(8, 8), 5, SKColors.Red));
            Assert.Equal(before, image.GetPixelSpan().ToArray());
        }

        [Fact]
        public void HardMask_MatchesSkia_ExceptPixelCentresOnTheOutline()
        {
            if (!Avx2ShapeFiller.IsSupported)
                return;

            var random = new Random(26);

            for (int shape = 0; shape < 40; shape++)
            {
                // Random polygons, some crossing the mask's borders, half of them with a second outline inside
                var outline = Star(new SKPoint(20 + (float)random.NextDouble() * 40, 15 + (float)random.NextDouble() * 30),
                    10 + (float)random.NextDouble() * 30, 4 + (float)random.NextDouble() * 8, 3 + random.Next(9));
                SKPoint[][] contours = shape % 2 == 0
                    ? [outline]
                    : [outline, Star(outline[0] + new SKPoint(-6, 4), 5, 2.5f, 5)];

                using var avx2 = MaskRasterizer.ToBitmap(contours, 3, 2, 72, 56, antiAlias: false);
                using var skia = SkiaSharpMask(contours, 3, 2, 72, 56);

                // A pixel is in the mask when its centre is inside. The two may only disagree on pixels whose centre lies on
                // the outline (within 0.05 px).
                for (int y = 0; y < 56; y++)
                {
                    for (int x = 0; x < 72; x++)
                    {
                        if (avx2.GetPixelSpan()[y * avx2.RowBytes + x] == skia.GetPixelSpan()[y * skia.RowBytes + x])
                            continue;

                        Assert.InRange(DistanceToOutline(new SKPoint(x + 3.5f, y + 2.5f), contours), 0, 0.05f);
                    }
                }
            }
        }

        [Fact]
        public void SmoothMask_MatchesTheDrawnMask()
        {
            if (!Avx2ShapeFiller.IsSupported)
                return;

            // A segmentation drawn in opaque black over white, and its smooth pixel mask: the same coverage, pixel for pixel
            var segmentation = new Segmentation
            {
                Label = new LabelModel { Index = 0, Name = "shape" },
                Confidence = 0.9,
                BoundingBox = new SKRectI(4, 3, 58, 44),
                Contours = [Star(new SKPoint(31.4f, 23.7f), 22.3f, 9.6f, 6), Rectangle(27.2f, 19.8f, 34.5f, 26.1f)]
            };

            using var image = CreateImage(64, 48, SKColors.White);
            image.Draw(new List<Segmentation> { segmentation }, new SegmentationDrawingOptions
            {
                BoundingBoxHexColors = ["#000000"],
                PixelMaskOpacity = 255,
                DrawBoundingBoxes = false,
                DrawLabels = false,
                DrawTrackedTail = false
            });

            using var mask = segmentation.GetPixelMask(64, 48, antiAliased: true);

            for (int y = 0; y < 48; y++)
                for (int x = 0; x < 64; x++)
                    Assert.Equal(255 - image.GetPixel(x, y).Red, mask.GetPixelSpan()[y * mask.RowBytes + x]);
        }

        [Fact]
        public void FillPolygons_ConcurrentCalls_GiveTheSameResult()
        {
            if (!Avx2ShapeFiller.IsSupported)
                return;

            SKPoint[][] small = [Star(new SKPoint(30, 25), 20, 8, 5)];
            SKPoint[][] large = [Rectangle(-50, -50, 400, 300)];   // uses the pooled buffers instead of the per-thread ones

            byte[] Draw(SKPoint[][] contours)
            {
                using var image = CreateImage(320, 240, Background);
                Avx2ShapeFiller.TryFillPolygons(image, contours, new SKColor(10, 220, 30, 140));
                Avx2ShapeFiller.TryFillCircle(image, new SKPoint(100, 80), 9.5f, new SKColor(250, 220, 0, 128));
                return image.GetPixelSpan().ToArray();
            }

            var expectedSmall = Draw(small);
            var expectedLarge = Draw(large);
            var results = new byte[64][];

            Parallel.For(0, results.Length, i => results[i] = Draw(i % 2 == 0 ? small : large));

            for (int i = 0; i < results.Length; i++)
                Assert.Equal(i % 2 == 0 ? expectedSmall : expectedLarge, results[i]);
        }

        #region Helpers
        private static SKBitmap CreateImage(int width, int height, SKColor color)
        {
            var image = new SKBitmap(new SKImageInfo(width, height, SKColorType.Bgra8888, SKAlphaType.Premul));
            image.Erase(color);
            return image;
        }

        private static SKPoint[] Rectangle(float left, float top, float right, float bottom)
            => [new(left, top), new(right, top), new(right, bottom), new(left, bottom)];

        private static SKPoint[] Star(SKPoint center, float outer, float inner, int points)
            => [.. Enumerable.Range(0, points * 2).Select(i =>
            {
                var radius = i % 2 == 0 ? outer : inner;
                var angle = Math.PI * i / points;
                return new SKPoint(center.X + radius * (float)Math.Cos(angle), center.Y + radius * (float)Math.Sin(angle));
            })];

        private static float Overlap(float a0, float a1, float b0, float b1)
            => Math.Max(0, Math.Min(a1, b1) - Math.Max(a0, b0));

        // Share of pixel (x, y) inside the circle, from 64 x 64 samples
        private static double CircleCoverage(int x, int y, SKPoint center, float radius)
        {
            const int samples = 64;
            int inside = 0;

            for (int sy = 0; sy < samples; sy++)
            {
                for (int sx = 0; sx < samples; sx++)
                {
                    float dx = x + (sx + 0.5f) / samples - center.X, dy = y + (sy + 0.5f) / samples - center.Y;

                    if (dx * dx + dy * dy <= radius * radius)
                        inside++;
                }
            }

            return inside / (double)(samples * samples);
        }

        private static float DistanceToOutline(SKPoint point, SKPoint[][] contours)
        {
            float best = float.MaxValue;

            foreach (var contour in contours)
            {
                for (int i = 0, j = contour.Length - 1; i < contour.Length; j = i++)
                {
                    var start = contour[j];
                    var edge = contour[i] - start;
                    float t = Math.Clamp(((point.X - start.X) * edge.X + (point.Y - start.Y) * edge.Y) / (edge.X * edge.X + edge.Y * edge.Y), 0, 1);
                    best = Math.Min(best, SKPoint.Distance(point, new SKPoint(start.X + t * edge.X, start.Y + t * edge.Y)));
                }
            }

            return best;
        }

        private static int MaxDifferenceFromSkia(int width, int height, SKPoint[][] contours, SKColor color)
        {
            using var avx2 = CreateImage(width, height, Background);
            Assert.True(Avx2ShapeFiller.TryFillPolygons(avx2, contours, color));

            using var skia = CreateImage(width, height, Background);
            using (var canvas = new SKCanvas(skia))
            using (var path = MaskRasterizer.CreatePath(contours))
            using (var paint = new SKPaint { Color = color, IsAntialias = true })
                canvas.DrawPath(path, paint);

            return MaxDifference(avx2, skia);
        }

        private static int MaxDifference(SKBitmap a, SKBitmap b)
        {
            var pa = a.GetPixelSpan();
            var pb = b.GetPixelSpan();
            int max = 0;

            for (int i = 0; i < pa.Length; i++)
                max = Math.Max(max, Math.Abs(pa[i] - pb[i]));

            return max;
        }

        // Sharp-edged mask drawn by SkiaSharp, for comparison
        private static SKBitmap SkiaSharpMask(SKPoint[][] contours, int left, int top, int width, int height)
        {
            var mask = new SKBitmap(new SKImageInfo(width, height, SKColorType.Alpha8, SKAlphaType.Premul));
            mask.Erase(SKColors.Transparent);

            using var canvas = new SKCanvas(mask);
            using var path = MaskRasterizer.CreatePath(contours);
            using var paint = new SKPaint { Color = SKColors.Black, IsAntialias = false };

            canvas.Translate(-left, -top);
            canvas.DrawPath(path, paint);
            return mask;
        }
        #endregion
    }
}
