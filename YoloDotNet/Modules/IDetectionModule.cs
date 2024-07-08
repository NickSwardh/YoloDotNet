namespace YoloDotNet.Modules
{
    public interface IDetectionModule : IDisposable
    {
        public event EventHandler VideoStatusEvent;
        public event EventHandler VideoProgressEvent;
        public event EventHandler VideoCompleteEvent;

        public OnnxModel OnnxModel { get; }
    }
}