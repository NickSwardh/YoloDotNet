namespace YoloDotNet.Modules.Interfaces
{
    internal interface IOBBDetectionModule : IModule
    {
        List<OBBDetection> ProcessImage(SKImage image, double confidence, double iou);
        Dictionary<int, List<OBBDetection>> ProcessVideo(VideoOptions options, double confidence, double iou);
    }
}