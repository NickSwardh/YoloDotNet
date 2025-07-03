namespace YoloDotNet.Modules.Interfaces
{
    public interface IObjectDetectionModule : IModule
    {
        List<ObjectDetection> ProcessImage<T>(T image, double confidence, double pixelConfidence,double iou);
    }
}
