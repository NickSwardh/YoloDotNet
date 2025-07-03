namespace YoloDotNet.Modules.Interfaces
{
    internal interface IOBBDetectionModule : IModule
    {
        List<OBBDetection> ProcessImage<T>(T image, double confidence, double pixelConfidence,double iou);
    }
}