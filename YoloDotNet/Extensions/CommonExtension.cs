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
        /// Calculates a new dynamic size based on the image dimensions and a given scale value.
        /// </summary>
        /// <param name="image">The image to calculate the dynamic size for.</param>
        /// <param name="scale">The initial scale value to be adjusted dynamically.</param>
        /// <returns>The new dynamically calculated size as a float value.</returns>
        public static float CalculateDynamicSize(this SKImage image, float scale)
        {
            // Calculate the scale factor based on the image resolution and a denominator
            float scaleFactor = image.Width / ImageConfig.SCALING_DENOMINATOR;

            var newSize = scale;

            newSize *= scaleFactor;

            return Math.Max(newSize, scale);
        }
    }
}
