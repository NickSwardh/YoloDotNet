namespace YoloDotNet.Modules.Interfaces
{
    internal interface IOBBDetectionModule : IModule
    {
        OBBDetection[] ProcessImage(SKImage image, double confidence, double pixelConfidence,double iou);
        Dictionary<int, OBBDetection[]> ProcessVideo(VideoOptions options, double confidence, double pixelConfidence,double iou);
    }
}