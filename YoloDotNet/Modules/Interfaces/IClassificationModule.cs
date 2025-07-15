namespace YoloDotNet.Modules.Interfaces
{
    public interface IClassificationModule : IModule
    {
        Classification[] ProcessImage(SKImage image, double classes, double pixelConfidence, double iou);
        Dictionary<int, Classification[]> ProcessVideo(VideoOptions options, double confidence, double pixelConfidence, double iou);
    }
}