namespace YoloDotNet.Modules.Interfaces
{
    public interface IClassificationModule : IModule
    {
        List<Classification> ProcessImage(SKImage image, double classes, double iou);
        Dictionary<int, List<Classification>> ProcessVideo(VideoOptions options, double confidence, double iou);
    }
}