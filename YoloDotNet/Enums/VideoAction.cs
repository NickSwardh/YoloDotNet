namespace YoloDotNet.Enums
{
    public enum VideoAction
    {
        // Get metadata from video
        GetMetaData,

        // Extract  metadata
        ExtractMetaData,

        // Extract audio file
        ExtractAudio,

        // Extract frames
        ExtractFrames,

        // Detecting objects
        ProcessFrames,

        // Compiling frames to video
        CompileFrames
    }
}
