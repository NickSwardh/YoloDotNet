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
    }
}
