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

            using var yolo = new Yolo(new YoloOptions
            {
                OnnxModel = model,
                Cuda = false,
            });

            using var image = SKBitmap.Decode(testImage);

            // Act
            var results = yolo.RunSegmentation(image);

            // Assert
            Assert.Equal(23, results.Count);
        }
    }
}
