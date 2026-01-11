// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Benchmarks
{
    internal class Program
    {
        static void Main(string[] args)
        {
#if DEBUG
            Console.WriteLine(
                "NOTE! You are running in DEBUG mode.\n" +
                "This mode is intended for development and debugging purposes only.\n" +
                "To obtain accurate and meaningful performance measurements, please run in RELEASE mode."
            );

            // Uncomment below code to run in debug during development.

            DefaultConfig.Instance.WithOptions(ConfigOptions.DisableOptimizationsValidator);

            var benchmark = new ObjectDetectionBenchmarks
            {
                YoloParam = YoloType.V8_Obj_GPU
            };

            benchmark.GlobalSetup();
            benchmark.ObjectDetection();
            benchmark.GlobalCleanup();
#else
            var config = DefaultConfig.Instance
                .WithOptions(ConfigOptions.DisableOptimizationsValidator)
                .WithSummaryStyle(
                    SummaryStyle.Default.WithRatioStyle(BenchmarkDotNet.Columns.RatioStyle.Trend)
                );

            BenchmarkSwitcher
                .FromAssembly(typeof(Program).Assembly)
                .Run(args, config);
#endif
        }
    }
}
