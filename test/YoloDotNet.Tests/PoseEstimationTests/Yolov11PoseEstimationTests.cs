namespace YoloDotNet.Tests.PoseEstimationTests
{
    public class Yolov11PoseEstimationTests
    {
        [Fact]
        public void RunPoseEstimation_Yolov11_GetExpectedNumberOfPoseEstimations_AssertTrue()
        {
            // Arrange
            var model = SharedConfig.GetTestModelV11(ModelType.PoseEstimation);
            var testImage = SharedConfig.GetTestImage(ImageType.Crosswalk);

            var yolo = new Yolo(new YoloOptions
            {
                OnnxModel = model,
                ModelType = ModelType.PoseEstimation,
                Cuda = false
            });

            var image = SKImage.FromEncodedData(testImage);

            // Act
            var results = yolo.RunPoseEstimation(image);

            // Assert
            Assert.Equal(10, results.Count);
        }
    }
}
