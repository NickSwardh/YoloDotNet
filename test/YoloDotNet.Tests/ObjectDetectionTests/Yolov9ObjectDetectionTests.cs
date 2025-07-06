namespace YoloDotNet.Tests.ObjectDetectionTests
{
    public class Yolov9ObjectDetectionTests
    {
        [Fact]
        public void RunObjectDetection_Yolov9_GetExpectedNumberOfObjects_AssertTrue()
        {
            // Arrange
            var model = SharedConfig.GetTestModelV9(ModelType.ObjectDetection);
            var testImage = SharedConfig.GetTestImage(ImageType.Street);

            using var yolo = new Yolo(new YoloOptions
            {
                OnnxModel = model,
                Cuda = false
            });

            using var image = SKBitmap.Decode(testImage);

            // Act
            var results = yolo.RunObjectDetection(image);

            // Assert
            Assert.Equal(28, results.Count);
        }
    }
}
