namespace YoloDotNet.Benchmarks
{
    using BenchmarkDotNet.Configs;
    using BenchmarkDotNet.Running;

    using YoloDotNet.Test.Common;
    using YoloDotNet.Benchmarks.ObjectDetectionTests;

    internal class Program
    {
        static void Main(string[] args)
        {
#if DEBUG
            //var simpleObjectDetectionTests = new SimpleObjectDetectionTests();
            //simpleObjectDetectionTests.GlobalSetup();
            //simpleObjectDetectionTests.RunSimpleObjectDetectionGpu();

            var resizeSourceObjectDetectionTests = new ResizeSourceObjectDetectionTests();
            resizeSourceObjectDetectionTests.GlobalSetup();

            var summary = BenchmarkRunner.Run<ObjectDetectionTests.SimpleObjectDetectionTests>(config:
                DefaultConfig.Instance
                .WithOptions(ConfigOptions.DisableOptimizationsValidator));
#else
            BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args);
#endif
        }
    }
}
