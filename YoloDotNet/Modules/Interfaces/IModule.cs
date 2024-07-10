namespace YoloDotNet.Modules.Interfaces
{
    public interface IModule : IDisposable
    {
        OnnxModel OnnxModel { get; }

        public event EventHandler VideoStatusEvent;
        public event EventHandler VideoProgressEvent;
        public event EventHandler VideoCompleteEvent;
    }
}
