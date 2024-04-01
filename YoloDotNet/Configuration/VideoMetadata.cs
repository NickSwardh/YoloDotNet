namespace YoloDotNet.Configuration
{
    public static class VideoMetadata
    {
        public static Dictionary<string, object> Tags => new()
        {
            { "title", "YoloDotNet" },
            { "comment", "Video frames analyzed using computer vision with YoloDotNet.\r\nGithub: https://github.com/NickSwardh/YoloDotNet" },
            { "genre", "Computer Vision" },
            { "date", DateTime.Now.Year }
        };
    }
}
