namespace YoloDotNet.Modules.Interfaces
{
    public interface ISegmentationModule : IModule
    {
        List<Segmentation> ProcessImage(SKBitmap image, double confidence, double pixelConfidence, double iou);
        //Dictionary<int, List<Segmentation>> ProcessVideo(VideoOptions options, double confidence, double pixelConfidence, double iou);
    }
}