using SkiaSharp;
using Xunit;
using YoloDotNet.Core;
using YoloDotNet.Extensions;
using YoloDotNet.Handlers;

namespace YoloDotNet.Tests.ImageResizeTests
{
    public class NormalizeMatchedTests
    {
        [Theory]
        [InlineData(SKColorType.Bgra8888, 16, 8)]
        [InlineData(SKColorType.Bgra8888, 13, 9)]
        [InlineData(SKColorType.Rgba8888, 16, 8)]
        [InlineData(SKColorType.Rgb888x, 13, 7)]
        public void DirectNormalize_MatchesCanvasPath(SKColorType colorType, int width, int height)
        {
            var mean = new[] { 0.485f, 0.456f, 0.406f };
            var std = new[] { 0.229f, 0.224f, 0.225f };
            using var bitmap = CreateBitmap(colorType, width, height, SKColorSpace.CreateSrgb());

            var expected = NormalizeWithCanvas(bitmap, mean, std);
            var actual = new float[expected.Length];
            var ok = bitmap.TryNormalizeMatched(width, height, 3, actual, out var size, mean: mean, std: std);

            Assert.True(ok);
            Assert.Equal(width, size.Width);
            Assert.Equal(height, size.Height);
            AssertPixelsClose(expected, actual);
        }

        [Fact]
        public void DirectNormalize_MatchesCanvasPath_ForRoi()
        {
            const int width = 20;
            const int height = 16;
            var roi = new SKRectI(2, 3, 15, 12);
            using var bitmap = CreateBitmap(SKColorType.Bgra8888, width, height, null);
            using var pinned = new PinnedMemoryBuffer(new SKImageInfo(roi.Width, roi.Height, SKColorType.Rgb888x, SKAlphaType.Opaque));
            bitmap.CopyImage(pinned, roi);

            var mean = new[] { 0.485f, 0.456f, 0.406f };
            var std = new[] { 0.229f, 0.224f, 0.225f };
            var expected = new float[roi.Width * roi.Height * 3];
            pinned.Pointer.NormalizePixelsToArray([1, 3, roi.Height, roi.Width], expected.Length, expected, mean, std);

            var actual = new float[expected.Length];
            var ok = bitmap.TryNormalizeMatched(roi.Width, roi.Height, 3, actual, out var size, roi, mean, std);

            Assert.True(ok);
            Assert.Equal(roi.Width, size.Width);
            Assert.Equal(roi.Height, size.Height);
            AssertPixelsClose(expected, actual);
        }

        [Fact]
        public void DirectNormalize_RejectsPremulAndMismatchedSize()
        {
            using var premul = new SKBitmap(new SKImageInfo(8, 8, SKColorType.Bgra8888, SKAlphaType.Premul));
            var destination = new float[8 * 8 * 3];
            Assert.False(premul.TryNormalizeMatched(8, 8, 3, destination, out _));

            using var bitmap = CreateBitmap(SKColorType.Bgra8888, 8, 8, null);
            Assert.False(bitmap.TryNormalizeMatched(4, 4, 3, destination, out _));
        }

        [Fact]
        public void SelectTopIndices_ReturnsLargestScoresDescending()
        {
            float[] scores = [-2f, 0.5f, 3f, -0.2f, 3f, 1.2f, -4f, 0.7f, 2.5f, -1f];
            var indices = new int[4];
            var selected = new float[4];

            var count = YoloCore.SelectTopIndices(scores, 4, indices, selected);

            Assert.Equal(4, count);
            Assert.Equal(scores.OrderByDescending(score => score).Take(4), selected);
            for (var i = 0; i < count; i++)
                Assert.Equal(scores[indices[i]], selected[i]);
        }

        [Fact]
        public void SelectTopIndices_MatchesFullSort_ForQueryClassMatrix()
        {
            const int queries = 300;
            const int labels = 20;
            var scores = new float[queries * labels];
            for (var i = 0; i < scores.Length; i++)
                scores[i] = (i * 17 % 1000) - 500;

            var indices = new int[queries];
            var selected = new float[queries];
            var count = YoloCore.SelectTopIndices(scores, queries, indices, selected);

            var expected = scores.OrderByDescending(score => score).Take(queries).ToArray();
            Assert.Equal(queries, count);
            Assert.Equal(expected, selected);
            for (var i = 0; i < count; i++)
                Assert.Equal(scores[indices[i]], selected[i]);
        }

        private static SKBitmap CreateBitmap(SKColorType colorType, int width, int height, SKColorSpace? colorSpace)
        {
            var bitmap = new SKBitmap(new SKImageInfo(width, height, colorType, SKAlphaType.Opaque, colorSpace));
            for (var y = 0; y < height; y++)
            {
                for (var x = 0; x < width; x++)
                    bitmap.SetPixel(x, y, new SKColor((byte)(x * 17), (byte)(y * 13), (byte)(x + y * 3), 255));
            }

            return bitmap;
        }

        private static void AssertPixelsClose(float[] expected, float[] actual)
        {
            Assert.Equal(expected.Length, actual.Length);
            for (var i = 0; i < expected.Length; i++)
            {
                var delta = MathF.Abs(expected[i] - actual[i]);
                Assert.True(delta <= 1e-5f, $"index {i}: expected {expected[i]}, actual {actual[i]}");
            }
        }

        private static float[] NormalizeWithCanvas(SKBitmap bitmap, float[] mean, float[] std)
        {
            using var pinned = new PinnedMemoryBuffer(new SKImageInfo(bitmap.Width, bitmap.Height, SKColorType.Rgb888x, SKAlphaType.Opaque));
            bitmap.CopyImage(pinned);
            var expected = new float[bitmap.Width * bitmap.Height * 3];
            pinned.Pointer.NormalizePixelsToArray([1, 3, bitmap.Height, bitmap.Width], expected.Length, expected, mean, std);
            return expected;
        }
    }
}
