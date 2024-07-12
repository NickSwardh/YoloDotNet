namespace YoloDotNet.Benchmarks.SkiaSharpTests
{
    [MemoryDiagnoser]
    public class AllSimpleTests
    {
        #region Test images

        private readonly string _bird = SharedConfig.GetTestImage(ImageType.Hummingbird);
        private readonly string _street = SharedConfig.GetTestImage(ImageType.Street);
        private readonly string _people = SharedConfig.GetTestImage(ImageType.People);
        private readonly string _crosswalk = SharedConfig.GetTestImage(ImageType.Crosswalk);
        private readonly string _island = SharedConfig.GetTestImage(ImageType.Island);

        #endregion

        #region Models

        private static string _classificationModel = SharedConfig.GetTestModelV8(ModelType.Classification);
        private static string _objectDetectionModel = SharedConfig.GetTestModelV8(ModelType.ObjectDetection);
        private static string _segmentationModel = SharedConfig.GetTestModelV8(ModelType.Segmentation);
        private static string _poseEstimationModel = SharedConfig.GetTestModelV8(ModelType.PoseEstimation);
        private static string _ObbDetectionModel = SharedConfig.GetTestModelV8(ModelType.ObbDetection);

        #endregion

        #region Tasks

        private SKImage _imageHummingbird;
        private SKImage _imageStreet;
        private SKImage _imagePeople;
        private SKImage _imageCrosswalk;
        private SKImage _imageIsland;

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
            _imageHummingbird = SKImage.FromEncodedData(_bird);
            _imageStreet = SKImage.FromEncodedData(_street);
            _imagePeople = SKImage.FromEncodedData(_people);
            _imageCrosswalk = SKImage.FromEncodedData(_crosswalk);
            _imageIsland = SKImage.FromEncodedData(_island);

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
            _imageHummingbird.Dispose();
            _imageStreet.Dispose();
            _imagePeople.Dispose();
            _imageCrosswalk.Dispose();
            _imageIsland.Dispose();

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
        public void ClassificationCpu()
        {
            _ = _yoloClassificationCPU.RunClassification(_imageHummingbird);
        }

        [Benchmark]
        public void ClassificationGpu()
        {
            _ = _yoloClassificationGPU.RunClassification(_imageHummingbird);
        }

        #endregion

        #region Test: Object Detection

        [Benchmark]
        public void ObjectDetectionCpu()
        {
            _ = _yoloObjedDetectionCPU.RunObjectDetection(_imageStreet);
        }

        [Benchmark]
        public void ObjectDetectionGpu()
        {
            _ = _yoloObjedDetectionGPU.RunObjectDetection(_imageStreet);
        }

        #endregion

        #region Test: Segmentation

        [Benchmark]
        public void SegmentationCpu()
        {
            _ = _yoloSegmentationCPU.RunSegmentation(_imagePeople);
        }

        [Benchmark]
        public void SegmentationGpu()
        {
            _ = _yoloSegmentationGPU.RunSegmentation(_imagePeople);
        }

        #endregion

        #region Test: Pose Estimation

        [Benchmark]
        public void PoseEstimationCPU()
        {
            _ = _yoloPoseEstimationCPU.RunPoseEstimation(_imageCrosswalk);
        }

        [Benchmark]
        public void PoseEstimationGpu()
        {
            _ = _yoloPoseEstimationGPU.RunPoseEstimation(_imageCrosswalk);
        }

        #endregion

        #region Test: OBB Detection

        [Benchmark]
        public void ObbDetectionCpu()
        {
            _ = _yoloObbDetectionCPU.RunObbDetection(_imageIsland);
        }

        [Benchmark]
        public void ObbDetectionGpu()
        {
            _ = _yoloObbDetectionGPU.RunObbDetection(_imageIsland);
        }

        #endregion
    }
}