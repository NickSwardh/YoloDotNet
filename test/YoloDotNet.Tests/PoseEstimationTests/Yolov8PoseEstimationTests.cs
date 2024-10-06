namespace YoloDotNet.Tests.PoseEstimationTests
{
    public class Yolov8PoseEstimationTests
    {
        [Fact]
        public void RunPoseEstimation_Yolov8_GetExpectedNumberOfPoseEstimations_AssertTrue()
        {
            // Arrange
            var model = SharedConfig.GetTestModelV8(ModelType.PoseEstimation);
            var testImage = SharedConfig.GetTestImage(ImageType.Crosswalk);

            var yolo = new Yolo(new YoloOptions
            {
                OnnxModel = model,
                ModelType = ModelType.PoseEstimation,
                ModelVersion = ModelVersion.V8,
                Cuda = false
            });

            var image = SKImage.FromEncodedData(testImage);

            // Act
            var results = yolo.RunPoseEstimation(image, 0.25, 0.45);

            // Assert
            Assert.Equal(10, results.Count);
        }
    }
}
