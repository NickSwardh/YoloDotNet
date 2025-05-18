namespace YoloDotNet.Modules.Interfaces
{
    public interface IClassificationModule : IModule
    {
        List<Classification> ProcessImage<T>(T image, double classes, double pixelConfidence, double iou);
    }
}