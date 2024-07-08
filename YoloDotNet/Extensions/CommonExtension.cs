namespace YoloDotNet.Extensions
{
    public static class CommonExtension
    {
        /// <summary>
        /// Converts a value to a string representation as a percentage with two decimal points.
        /// </summary>
        /// <param name="value">The value to convert.</param>
        /// <returns>A string representation of the percentage with two decimal points.</returns>
        public static string ToPercent(this double value)
            => (value * 100).ToString("0.##", CultureInfo.InvariantCulture);

        /// <summary>
        /// Calculates new font size and bounding box border thickness based on the image dimensions.
        /// </summary>
        /// <param name="image">The image to calculate font size for.</param>
        /// <returns>a float tuple with new font size and bounding box border thickness.</returns>
        public static (float, float) CalculateFontSize(this SKImage image)
        {
            // Calculate the scale factor based on the image resolution
            float scaleFactor = image.Width / 1280; // adjust the denominator to your desired resolution

            var fontSize = ImageConfig.FONT_SIZE;
            var strokeWidth = ImageConfig.BORDER_THICKNESS;

            fontSize *= scaleFactor;
            fontSize = Math.Max(fontSize, ImageConfig.FONT_SIZE);

            strokeWidth *= scaleFactor;

            return (fontSize, strokeWidth);
        }
    }
}
