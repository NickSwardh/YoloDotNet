namespace YoloDotNet.Enums
{
    /// <summary>
    /// Strongly typed names for video pipeline stages.
    /// </summary>
    public enum VideoAction
    {
        GetMetaData,
        ExtractMetaData,
        ExtractAudio,
        ExtractFrames,
        ProcessFrames,
        CompileFrames
    }
}
