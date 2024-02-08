namespace YoloDotNet.Tests.Configuration
{
    public static class Config
    {
        const string BASE_MODELS = @"..\..\..\assets\models";
        const string BASE_MEDIA = @"..\..\..\assets\media";

        public static string GetTestModel(ModelType modelType) => modelType switch
        {
            ModelType.Classification => Path.Combine(BASE_MODELS, "yolov8s-cls.onnx"),
            ModelType.ObjectDetection => Path.Combine(BASE_MODELS, "yolov8s.onnx"),
            ModelType.Segmentation => Path.Combine(BASE_MODELS, "yolov8s-seg.onnx"),
            _ => throw new ArgumentException("Unknown modeltype.")
        };

        public static string GetTestImage(ImageType imageType) => imageType switch
        {
            ImageType.Hummingbird => Path.Combine(BASE_MEDIA, "hummingbird.jpg"),
            ImageType.Street => Path.Combine(BASE_MEDIA, "street.jpg"),
            ImageType.People => Path.Combine(BASE_MEDIA, "people.jpg"),
            _ => throw new ArgumentException("Unknown modeltype.")
        };
    }
}
