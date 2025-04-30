namespace YoloDotNet.Modules.Interfaces
{
    internal interface IPoseEstimationModule : IModule
    {
        List<PoseEstimation> ProcessImage(SKBitmap image, double confidence, double pixelConfidence, double iou);
        //Dictionary<int, List<PoseEstimation>> ProcessVideo(VideoOptions options, double confidence, double pixelConfidence, double iou);
    }
}