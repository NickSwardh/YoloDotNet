namespace YoloDotNet.Tests.SegmentationTests
{
    public class Yolov8SegmentationTests
    {
        [Fact]
        public void RunSegmentation_Yolov8_GetExpectedNumberOfSegmentations_AssertTrue()
        {
            // Arrange
            var model = SharedConfig.GetTestModelV8(ModelType.Segmentation);
            var testImage = SharedConfig.GetTestImage(ImageType.People);

            var yolo = new Yolo(new YoloOptions
            {
                OnnxModel = model,
                ModelType = ModelType.Segmentation,
                ModelVersion = ModelVersion.V8,
                Cuda = false
            });

            var image = SKImage.FromEncodedData(testImage);

            // Act
            var results = yolo.RunSegmentation(image, 0.25, 0.45);

            // Assert
            Assert.Equal(20, results.Count);
        }
    }
}
