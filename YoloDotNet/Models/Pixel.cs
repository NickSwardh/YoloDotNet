namespace YoloDotNet.Models
{
    /// <summary>
    /// Pixel location and confidence value
    /// </summary>
    /// <param name="X"></param>
    /// <param name="Y"></param>
    /// <param name="Confidence"></param>
    public record Pixel(int X, int Y, double Confidence);
}
