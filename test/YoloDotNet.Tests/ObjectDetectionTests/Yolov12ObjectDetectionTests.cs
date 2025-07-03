namespace YoloDotNet.Tests.ObjectDetectionTests
{
    public class Yolov12ObjectDetectionTests
    {
        [Fact]
        public void RunObjectDetection_Yolov12_GetExpectedNumberOfObjects_AssertTrue()
        {
            // Arrange
            var model = SharedConfig.GetTestModelV12(ModelType.ObjectDetection);
            var testImage = SharedConfig.GetTestImage(ImageType.Street);

            var yolo = new Yolo(new YoloOptions
            {
                OnnxModel = model,
                Cuda = false
            });

            using var image = SKBitmap.Decode(testImage);

            // Act
            var results = yolo.RunObjectDetection(image);

            // Assert
            Assert.Equal(31, results.Count);
        }
    }
}
