namespace YoloDotNet.Modules.Interfaces
{
    public interface IObjectDetectionModule : IModule
    {
        List<ObjectDetection> ProcessImage(SKBitmap image, double confidence, double pixelConfidence,double iou);
    }
}
