namespace YoloDotNet.Benchmarks.SkiaSharpTests
{
    [MemoryDiagnoser]
    public class Load_Image_And_Run_Inference_All
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
            _yoloClassificationGPU = new Yolo(new YoloOptions
            {
                OnnxModel = _classificationModel,
                ModelType = ModelType.Classification,
                Cuda = true
            });

            _yoloClassificationCPU = new Yolo(new YoloOptions
            {
                OnnxModel = _classificationModel,
                ModelType = ModelType.Classification,
                Cuda = false
            });

            _yoloObjedDetectionGPU = new Yolo(new YoloOptions
            {
                OnnxModel = _objectDetectionModel,
                ModelType = ModelType.ObjectDetection,
                Cuda = true
            });
            _yoloObjedDetectionCPU = new Yolo(new YoloOptions
            {
                OnnxModel = _objectDetectionModel,
                ModelType = ModelType.ObjectDetection,
                Cuda = false
            });

            _yoloSegmentationGPU = new Yolo(new YoloOptions
            {
                OnnxModel = _segmentationModel,
                ModelType = ModelType.Segmentation,
                Cuda = true
            });
            _yoloSegmentationCPU = new Yolo(new YoloOptions
            {
                OnnxModel = _segmentationModel,
                ModelType = ModelType.Segmentation,
                Cuda = false
            });

            _yoloPoseEstimationGPU = new Yolo(new YoloOptions
            {
                OnnxModel = _poseEstimationModel,
                ModelType = ModelType.PoseEstimation,
                Cuda = true
            });
            _yoloPoseEstimationCPU = new Yolo(new YoloOptions
            {
                OnnxModel = _poseEstimationModel,
                ModelType = ModelType.PoseEstimation,
                Cuda = false
            });

            _yoloObbDetectionGPU = new Yolo(new YoloOptions
            {
                OnnxModel = _ObbDetectionModel,
                ModelType = ModelType.ObbDetection,
                Cuda = true
            });
            _yoloObbDetectionCPU = new Yolo(new YoloOptions
            {
                OnnxModel = _ObbDetectionModel,
                ModelType = ModelType.ObbDetection,
                Cuda = false
            });
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
        public void ClassificationCPU()
        {
            using var image = SKImage.FromEncodedData(_bird);
            _ = _yoloClassificationCPU.RunClassification(image);
        }

        [Benchmark]
        public void ClassificationGPU()
        {
            using var image = SKImage.FromEncodedData(_bird);
            _ = _yoloClassificationGPU.RunClassification(image);
        }

        #endregion

        #region Test: Object Detection

        [Benchmark]
        public void ObjectDetectionCPU()
        {
            using var image = SKImage.FromEncodedData(_street);
            _ = _yoloObjedDetectionCPU.RunObjectDetection(image);
        }

        [Benchmark]
        public void ObjectDetectionGPU()
        {
            using var image = SKImage.FromEncodedData(_street);
            _ = _yoloObjedDetectionGPU.RunObjectDetection(image);
        }

        #endregion

        #region Test: Segmentation

        [Benchmark]
        public void SegmentationCPU()
        {
            using var image = SKImage.FromEncodedData(_people);
            _ = _yoloSegmentationCPU.RunSegmentation(image);
        }

        [Benchmark]
        public void SegmentationGPU()
        {
            using var image = SKImage.FromEncodedData(_people);
            _ = _yoloSegmentationGPU.RunSegmentation(image);
        }

        #endregion

        #region Test: Pose Estimation

        [Benchmark]
        public void PoseEstimationCPU()
        {
            using var image = SKImage.FromEncodedData(_crosswalk);
            _ = _yoloPoseEstimationCPU.RunPoseEstimation(image);
        }

        [Benchmark]
        public void PoseEstimationGPU()
        {
            using var image = SKImage.FromEncodedData(_crosswalk);
            _ = _yoloPoseEstimationGPU.RunPoseEstimation(image);
        }

        #endregion

        #region Test: OBB Detection

        [Benchmark]
        public void ObbDetectionCPU()
        {
            using var image = SKImage.FromEncodedData(_island);
            _ = _yoloObbDetectionCPU.RunObbDetection(image);
        }

        [Benchmark]
        public void ObbDetectionGPU()
        {
            using var image = SKImage.FromEncodedData(_island);
            _ = _yoloObbDetectionGPU.RunObbDetection(image);
        }

        #endregion
    }
}
