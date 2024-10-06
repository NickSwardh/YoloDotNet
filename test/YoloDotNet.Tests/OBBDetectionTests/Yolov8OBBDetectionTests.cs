namespace YoloDotNet.Tests.OBBDetectionTests
{
    public class Yolov8OBBDetectionTests
    {
        [Fact]
        public void RunObbDetection_Yolov8_GetExpectedNumberOfObjects_AssertTrue()
        {
            // Arrange
            var model = SharedConfig.GetTestModelV8(ModelType.ObbDetection);
            var testImage = SharedConfig.GetTestImage(ImageType.Island);

            var yolo = new Yolo(new YoloOptions
            {
                OnnxModel = model,
                ModelType = ModelType.ObbDetection,
                ModelVersion = ModelVersion.V8,
                Cuda = false
            });

            var image = SKImage.FromEncodedData(testImage);

            // Act
            var results = yolo.RunObbDetection(image, 0.25, 0.45);

            // Assert
            Assert.Equal(5, results.Count);
        }
    }
}
