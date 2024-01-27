using System.Globalization;

namespace YoloDotNet.Extensions
{
    public static class CommonExtension
    {
        /// <summary>
        /// Calculate confidence score in percent
        /// </summary>
        /// <param name="confidence"></param>
        /// <returns></returns>
        public static string ToPercent(this double confidence)
            => (confidence * 100).ToString("0.##", CultureInfo.InvariantCulture);
        /// <summary>
        /// Calculates new font size based on image DPI (dots per inch), image dimensions and the provided font size.
        /// If no DPI information is present in the image metadata, a default DPI of 72 is assumed.
        /// </summary>
        /// <param name="image">The image to calculate font size for.</param>
        /// <param name="fontSize">The original font size to be adjusted.</param>
        /// <returns>The adjusted font size based on DPI and image dimensions.</returns>
        /// <exception cref="ArgumentNullException">Thrown when the input image is null.</exception>
        public static float CalculateFontSizeByDpi(this Image image, float fontSize)
        {
            ArgumentNullException.ThrowIfNull(image);

            var dpi = 72f; // Assume default if no DPI info is present in image metadata

            if (image.Metadata?.VerticalResolution > 1)
                dpi = (float)image.Metadata.VerticalResolution;

            // Font size * dpi
            var scale = fontSize * dpi;

            // Get smallest ratio
            var ratio = Math.Min(image.Width / scale, image.Height / scale);

            // Calculate adjusted font size
            var newFontSize = fontSize * ratio;

            return newFontSize > fontSize ? newFontSize : fontSize;
        }
    }
}
