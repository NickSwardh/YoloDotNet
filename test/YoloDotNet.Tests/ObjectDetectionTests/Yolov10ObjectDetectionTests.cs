namespace YoloDotNet.Tests.ObjectDetectionTests
{
    public class Yolov10ObjectDetectionTests
    {
        [Fact]
        public void RunObjectDetection_Yolov10_GetExpectedNumberOfObjects_AssertTrue()
        {
            // Arrange
            var model = SharedConfig.GetTestModelV10(ModelType.ObjectDetection);
            var testImage = SharedConfig.GetTestImage(ImageType.Street);

            var yolo = new Yolo(new YoloOptions
            {
                OnnxModel = model,
                ModelType = ModelType.ObjectDetection,
                ModelVersion = ModelVersion.V10,
                Cuda = false
            });

            var image = SKImage.FromEncodedData(testImage);

            // Act
            var results = yolo.RunObjectDetection(image, 0.25, 0.45);

            // Assert
            Assert.Equal(29, results.Count);
        }
    }
}
