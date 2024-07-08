namespace YoloDotNet.Modules
{
    public interface IModule<TImage, TVideo>
    {
        TImage ProcessImage(SKImage image, double confidence, double iou);
        TVideo ProcessVideo(VideoOptions options, double confidence, double iou);
    }
}
