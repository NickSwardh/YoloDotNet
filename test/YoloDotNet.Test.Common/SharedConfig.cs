namespace YoloDotNet.Test.Common
{
    using System;
    using System.IO;

    using YoloDotNet.Enums;
    using YoloDotNet.Test.Common.Enums;

    public static class SharedConfig
    {
        private const string ASSETS_FOLDER = @".\assets";
        private const string BASE_MODELS = ASSETS_FOLDER + @"\models";
        private const string BASE_MEDIA = ASSETS_FOLDER + @"\media";

        public static string GetTestModel(ModelType modelType) => modelType switch
        {
            ModelType.Classification => Path.Combine(BASE_MODELS, "yolov8s-cls.onnx"),
            ModelType.ObjectDetection => Path.Combine(BASE_MODELS, "yolov8s.onnx"),
            ModelType.ObbDetection => Path.Combine(BASE_MODELS, "yolov8s-obb.onnx"),
            ModelType.Segmentation => Path.Combine(BASE_MODELS, "yolov8s-seg.onnx"),
            ModelType.PoseEstimation => Path.Combine(BASE_MODELS, "yolov8s-pose.onnx"),
            _ => throw new ArgumentException("Unknown modeltype.")
        };

        public static string GetTestImage(ImageType imageType) => imageType switch
        {
            ImageType.Hummingbird => Path.Combine(BASE_MEDIA, "hummingbird.jpg"),
            ImageType.Street => Path.Combine(BASE_MEDIA, "street.jpg"),
            ImageType.People => Path.Combine(BASE_MEDIA, "people.jpg"),
            ImageType.Crosswalk => Path.Combine(BASE_MEDIA, "crosswalk.jpg"),
            ImageType.Island => Path.Combine(BASE_MEDIA, "island.jpg"),
            _ => throw new ArgumentException("Unknown modeltype.")
        };

        public static string GetTestImage(string imageName) => Path.Combine(BASE_MEDIA, imageName);
    }
}
