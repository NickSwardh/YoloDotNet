namespace YoloDotNet.Modules.Interfaces
{
    public interface IClassificationModule : IModule
    {
        List<Classification> ProcessImage(SKImage image, double classes, double pixelConfidence, double iou);
        Dictionary<int, List<Classification>> ProcessVideo(VideoOptions options, double confidence, double pixelConfidence, double iou);
    }
}