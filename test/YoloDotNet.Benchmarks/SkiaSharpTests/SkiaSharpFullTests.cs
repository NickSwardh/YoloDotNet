using System.Reflection.Metadata;

namespace YoloDotNet.Benchmarks.SkiaSharpTests
{

    [MemoryDiagnoser]
    public class SkiaSharpFullTests
    {
        #region Test images

        private readonly string _bird = SharedConfig.GetTestImage(ImageType.Hummingbird);
        private readonly string _street = SharedConfig.GetTestImage(ImageType.Street);
        private readonly string _people = SharedConfig.GetTestImage(ImageType.People);
        private readonly string _crosswalk = SharedConfig.GetTestImage(ImageType.Crosswalk);
        private readonly string _island = SharedConfig.GetTestImage(ImageType.Island);

        #endregion

        #region Models

        private static string _classificationModel = SharedConfig.GetTestModel(ModelType.Classification);
        private static string _objectDetectionModel = SharedConfig.GetTestModel(ModelType.ObjectDetection);
        private static string _segmentationModel = SharedConfig.GetTestModel(ModelType.Segmentation);
        private static string _poseEstimationModel = SharedConfig.GetTestModel(ModelType.PoseEstimation);
        private static string _ObbDetectionModel = SharedConfig.GetTestModel(ModelType.ObbDetection);

        #endregion

        #region Tasks

        private Yolo _yoloClassificationCPU;
        private Yolo _yoloClassificationGPU;

        private Yolo _yoloObjedDetectionCPU;
        private Yolo _yoloObjedDetectionGPU;

        private Yolo _yoloSegmentationCPU;
        private Yolo _yoloSegmentationGPU;

        private Yolo _yoloPoseEstimationCPU;
        private Yolo _yoloPoseEstimationGPU;

        private Yolo _yoloObbDetectionCPU;
        private Yolo _yoloObbDetectionGPU;

        #endregion

        #region Setup

        [GlobalSetup]
        public void Setup()
        {
            _yoloClassificationCPU = new Yolo(_classificationModel, false);
            _yoloClassificationGPU = new Yolo(_classificationModel, true);

            _yoloObjedDetectionCPU = new Yolo(_objectDetectionModel, false);
            _yoloObjedDetectionGPU = new Yolo(_objectDetectionModel, true);

            _yoloSegmentationCPU = new Yolo(_segmentationModel, false);
            _yoloSegmentationGPU = new Yolo(_segmentationModel, true);

            _yoloPoseEstimationCPU = new Yolo(_poseEstimationModel, false);
            _yoloPoseEstimationGPU = new Yolo(_poseEstimationModel, true);

            _yoloObbDetectionCPU = new Yolo(_ObbDetectionModel, false);
            _yoloObbDetectionGPU = new Yolo(_ObbDetectionModel, true);
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            _yoloClassificationCPU.Dispose();
            _yoloClassificationGPU.Dispose();

            _yoloObjedDetectionCPU.Dispose();
            _yoloObjedDetectionGPU.Dispose();

            _yoloSegmentationCPU.Dispose();
            _yoloSegmentationGPU.Dispose();

            _yoloPoseEstimationCPU.Dispose();
            _yoloPoseEstimationGPU.Dispose();

            _yoloObbDetectionCPU.Dispose();
            _yoloObbDetectionGPU.Dispose();
        }

        #endregion

        #region Test: Classification

        [Benchmark]
        public void FullClassificationCPU()
        {
            using var image = SKImage.FromEncodedData(_bird);
            _ = _yoloClassificationCPU.RunClassification(image);
        }

        [Benchmark]
        public void FullClassificationGPU()
        {
            using var image = SKImage.FromEncodedData(_bird);
            _ = _yoloClassificationGPU.RunClassification(image);
        }

        #endregion

        #region Test: Object Detection

        [Benchmark]
        public void FullObjectDetectionCPU()
        {
            using var image = SKImage.FromEncodedData(_street);
            _ = _yoloObjedDetectionCPU.RunObjectDetection(image);
        }

        [Benchmark]
        public void FullObjectDetectionGPU()
        {
            using var image = SKImage.FromEncodedData(_street);
            _ = _yoloObjedDetectionGPU.RunObjectDetection(image);
        }

        #endregion

        #region Test: Segmentation

        [Benchmark]
        public void FullSegmentationCPU()
        {
            using var image = SKImage.FromEncodedData(_people);
            _ = _yoloSegmentationCPU.RunSegmentation(image);
        }

        [Benchmark]
        public void FullSegmentationGPU()
        {
            using var image = SKImage.FromEncodedData(_people);
            _ = _yoloSegmentationGPU.RunSegmentation(image);
        }

        #endregion

        #region Test: Pose Estimation

        [Benchmark]
        public void FullPoseEstimationCPU()
        {
            using var image = SKImage.FromEncodedData(_street);
            _ = _yoloPoseEstimationCPU.RunPoseEstimation(image);
        }

        [Benchmark]
        public void FullPoseEstimationGPU()
        {
            using var image = SKImage.FromEncodedData(_street);
            _ = _yoloPoseEstimationGPU.RunPoseEstimation(image);
        }

        #endregion

        #region Test: OBB Detection

        [Benchmark]
        public void FullObbDetectionCPU()
        {
            using var image = SKImage.FromEncodedData(_street);
            _ = _yoloObbDetectionCPU.RunObbDetection(image);
        }

        [Benchmark]
        public void FullObbDetectionGPU()
        {
            using var image = SKImage.FromEncodedData(_street);
            _ = _yoloObbDetectionGPU.RunObbDetection(image);
        }

        #endregion
    }
}
