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
        public static float CalculateDynamicSize(this SKBitmap image, float scale)
        {
            // Calculate the scale factor based on the image resolution and a denominator
            float scaleFactor = image.Width / ImageConfig.SCALING_DENOMINATOR;

            var newSize = scale;

            newSize *= scaleFactor;

            return Math.Max(newSize, scale);
        }

        /// <summary>
        /// Filters a list of object detection results, keeping only the objects whose labels match the specified filter classes.
        /// </summary>
        /// <typeparam name="T">The type of the detection results. Must be one of the supported types.</typeparam>
        /// <param name="result">The list of detection results to filter.</param>
        /// <param name="filterClasses">A set of class labels to retain in the filtered results.</param>
        /// <returns>A filtered list containing only detection results where the label matches any of the specified filter classes.</returns>
        /// <exception cref="ArgumentException">Thrown if the type <typeparamref name="T"/> is not a supported detection type.</exception>
        public static List<T> FilterLabels<T>(this List<T> result, HashSet<string> filterClasses)
            => [.. typeof(T) switch
            {
                Type t when t == typeof(Classification) => result.Cast<Classification>().Where(x => filterClasses.Contains(x.Label)).Cast<T>(),
                Type t when t == typeof(ObjectDetection) => result.Cast<ObjectDetection>().Where(x => filterClasses.Contains(x.Label.Name)).Cast<T>(),
                Type t when t == typeof(OBBDetection) => result.Cast<OBBDetection>().Where(x => filterClasses.Contains(x.Label.Name)).Cast<T>(),
                Type t when t == typeof(PoseEstimation) => result.Cast<PoseEstimation>().Where(x => filterClasses.Contains(x.Label.Name)).Cast<T>(),
                Type t when t == typeof(Segmentation) => result.Cast<Segmentation>().Where(x => filterClasses.Contains(x.Label.Name)).Cast<T>(),
                _ => throw new ArgumentException("Invalid type", nameof(result))
            }];
    }
}
