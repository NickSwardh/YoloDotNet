namespace YoloDotNet.Modules
{
    internal class ModuleFactory
    {
        public static IDetectionModule CreateModule(YoloOptions options)
        {
            var onnxModel = options.OnnxModel;
            var modelType = options.ModelType;
            var cuda = options.Cuda;
            var primeGpu = options.PrimeGpu;
            var gpuId = options.GpuId;

            return modelType switch
            {
                ModelType.Classification => new ClassificationModule(onnxModel, cuda, primeGpu, gpuId),
                ModelType.ObjectDetection => new ObjectDetectionModule(onnxModel, cuda, primeGpu, gpuId),
                ModelType.ObbDetection => new ObbDetectionModule(onnxModel, cuda, primeGpu, gpuId),
                ModelType.Segmentation => new SegmentationModule(onnxModel, cuda, primeGpu, gpuId),
                ModelType.PoseEstimation => new PoseEstimationModule(onnxModel, cuda, primeGpu, gpuId),
                _ => throw new InvalidOperationException("Unsupported detection type")
            };
        }
    }
}
