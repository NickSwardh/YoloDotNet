namespace YoloDotNet.Modules.Interfaces
{
    public interface ISegmentationModule : IModule
    {
        List<Segmentation> ProcessImage<T>(T image, double confidence, double pixelConfidence, double iou);
    }
}