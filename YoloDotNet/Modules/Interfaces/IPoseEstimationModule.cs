namespace YoloDotNet.Modules.Interfaces
{
    internal interface IPoseEstimationModule : IModule
    {
        PoseEstimation[] ProcessImage(SKImage image, double confidence, double pixelConfidence, double iou);
        Dictionary<int, PoseEstimation[]> ProcessVideo(VideoOptions options, double confidence, double pixelConfidence, double iou);
    }
}