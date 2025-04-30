namespace YoloDotNet.Modules.Interfaces
{
    public interface IModule : IDisposable
    {
        OnnxModel OnnxModel { get; }
    }
}
