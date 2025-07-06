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

            using var yolo = new Yolo(new YoloOptions
            {
                OnnxModel = model,
                Cuda = false
            });

            using var image = SKBitmap.Decode(testImage);

            // Act
            var results = yolo.RunObjectDetection(image);

            // Assert
            Assert.Equal(29, results.Count);
        }
    }
}
