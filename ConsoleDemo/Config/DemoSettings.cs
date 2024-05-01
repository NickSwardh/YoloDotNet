namespace ConsoleDemo.Config
{
    // Models and test data from assets in YoloDotNet.Tests
    static class DemoSettings
    {
        public const string ASSETS_FOLDER = @"..\..\..\..\YoloDotNet.Tests\Assets";
        public const string MODELS_FOLDER = ASSETS_FOLDER + @"\models";
        public const string MEDIA_FOLDER = ASSETS_FOLDER + @"\media";
        public static readonly string OUTPUT_FOLDER = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "YoloDotNet_Results");
    }
}
