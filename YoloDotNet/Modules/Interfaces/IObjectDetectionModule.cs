namespace YoloDotNet.Modules.Interfaces
{
    public interface IObjectDetectionModule : IModule
    {
        List<ObjectDetection> ProcessImage(SKImage image, double confidence, double iou);
        Dictionary<int, List<ObjectDetection>> ProcessVideo(VideoOptions options, double confidence, double iou);
    }
}
