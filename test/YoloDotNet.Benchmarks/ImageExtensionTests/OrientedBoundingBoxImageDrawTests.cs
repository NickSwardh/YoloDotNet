namespace YoloDotNet.Benchmarks.ImageExtensionTests
{
    [MemoryDiagnoser]
    public class OrientedBoundingBoxImageDrawTests
    {
        #region Fields

        private static string _model = SharedConfig.GetTestModel(modelType: ModelType.ObbDetection);
        private static string _testImage = SharedConfig.GetTestImage(imageType: ImageType.Island);

        private Yolo _cpuYolo;
        private Image _image;
        private List<OBBDetection> _oBBDetections;

        #endregion Fields

        #region Methods

        [GlobalSetup]
        public void GlobalSetup()
        {
            _cpuYolo = new Yolo(onnxModel: _model, cuda: false);
            _image = Image.Load(path: _testImage);
            _oBBDetections = _cpuYolo.RunObbDetection(img: _image, confidence: 0.25, iou: 0.45);
        }

        [Params(true,false)]
        public bool DrawConfidence { get; set; }

        [Benchmark]
        public Image DrawOrientedBoundingBox()
        {
            _image.Draw(detections: _oBBDetections, drawConfidence: DrawConfidence);

            return _image;
        }

        #endregion Methods
    }
}
