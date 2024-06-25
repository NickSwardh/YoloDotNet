using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Reports;
using BenchmarkDotNet.Running;

namespace YoloDotNet.Benchmarks
{
    internal class Program
    {
        static void Main(string[] args)
        {
#if DEBUG
            //var imageDrawTests = new ObjectDetectionImageDrawTests();
            //imageDrawTests.GlobalSetup();
            //imageDrawTests.IterationSetup();
            //imageDrawTests.DrawObjectDetection();

            //var simpleObjectDetectionTests = new SimpleObjectDetectionTests();
            //simpleObjectDetectionTests.GlobalSetup();
            //simpleObjectDetectionTests.RunSimpleObjectDetectionGpu();

            //var resizeSourceObjectDetectionTests = new ResizeSourceObjectDetectionTests();
            //resizeSourceObjectDetectionTests.GlobalSetup();

            var summary = BenchmarkRunner.Run<ObjectDetectionTests.SimpleObjectDetectionTests>(config:
                DefaultConfig.Instance
                .WithOptions(ConfigOptions.DisableOptimizationsValidator));
#else
            BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args, DefaultConfig.Instance
                .WithOptions(ConfigOptions.DisableOptimizationsValidator)
                .WithSummaryStyle(summaryStyle: SummaryStyle.Default.WithRatioStyle(ratioStyle: BenchmarkDotNet.Columns.RatioStyle.Trend)));
#endif
        }
    }
}
