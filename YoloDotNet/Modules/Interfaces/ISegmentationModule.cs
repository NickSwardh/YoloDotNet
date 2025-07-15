namespace YoloDotNet.Modules.Interfaces
{
    public interface ISegmentationModule : IModule
    {
        Segmentation[] ProcessImage(SKImage image, double confidence, double pixelConfidence, double iou);
        Dictionary<int, Segmentation[]> ProcessVideo(VideoOptions options, double confidence, double pixelConfidence, double iou);
    }
}