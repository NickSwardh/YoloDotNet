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

            using var yolo = new Yolo(new YoloOptions
            {
                OnnxModel = model,
                Cuda = false
            });

            using var image = SKBitmap.Decode(testImage);

            // Act
            var results = yolo.RunPoseEstimation(image, 0.25, 0.45);

            // Assert
            Assert.Equal(10, results.Count);
        }
    }
}
