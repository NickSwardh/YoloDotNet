namespace YoloDotNet.Modules.Interfaces
{
    internal interface IPoseEstimationModule : IModule
    {
        List<PoseEstimation> ProcessImage<T>(T image, double confidence, double pixelConfidence, double iou);
    }
}