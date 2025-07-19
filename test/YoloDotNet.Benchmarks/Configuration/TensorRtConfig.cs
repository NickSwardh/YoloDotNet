namespace YoloDotNet.Benchmarks.Configuration
{
    /// <summary>
    /// Defines the absolute path to the directory where TensorRT engine cache files are stored or loaded.
    ///
    /// Engine caching avoids re-building the TensorRT engine on every inference session, significantly 
    /// improving performance for repeated runs. Ensure this folder exists and is writable. 
    ///
    /// 📌 Update this path to a valid local directory on your machine before running any TensorRT benchmarks.
    /// </summary>
    public static class TensorRtConfig
    {
        public const string TRT_ENGINE_CACHE_PATH = @"C:\Temp\YoloDotNet_Engine_Cache";
    }
}
