namespace YoloDotNet.Tests.ObjectDetectionTests
{
    public class Yolov11ObjectDetectionTests
    {
        [Fact]
        public void RunObjectDetection_Yolov11_GetExpectedNumberOfObjects_AssertTrue()
        {
            // Arrange
            var model = SharedConfig.GetTestModelV11(ModelType.ObjectDetection);
            var testImage = SharedConfig.GetTestImage(ImageType.Street);

            var yolo = new Yolo(new YoloOptions
            {
                OnnxModel = model,
                ModelType = ModelType.ObjectDetection,
                Cuda = false
            });

            using var image = SKBitmap.Decode(testImage);

            // Act
            var results = yolo.RunObjectDetection(image);

            // Assert
            Assert.Equal(30, results.Count);
        }
    }
}
