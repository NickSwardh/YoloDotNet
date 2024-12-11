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

            var yolo = new Yolo(new YoloOptions
            {
                OnnxModel = model,
                ModelType = ModelType.ObjectDetection,
                HwAccelerator = HwAcceleratorType.None
            });

            var image = SKImage.FromEncodedData(testImage);

            // Act
            var results = yolo.RunObjectDetection(image, 0.25, 0.45);

            // Assert
            Assert.Equal(25, results.Count);
        }
    }
}
