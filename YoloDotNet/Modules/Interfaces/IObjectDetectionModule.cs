namespace YoloDotNet.Modules.Interfaces
{
    public interface IObjectDetectionModule : IModule
    {
        ObjectDetection[] ProcessImage(SKImage image, double confidence, double pixelConfidence,double iou);
        Dictionary<int, ObjectDetection[]> ProcessVideo(VideoOptions options, double confidence, double pixelConfidence, double iou);
    }
}
