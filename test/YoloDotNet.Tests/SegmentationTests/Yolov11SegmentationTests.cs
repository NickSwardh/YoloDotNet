namespace YoloDotNet.Tests.SegmentationTests
{
    public class Yolov11SegmentationTests
    {
        [Fact]
        public void RunSegmentation_Yolov11_GetExpectedNumberOfSegmentations_AssertTrue()
        {
            // Arrange
            var model = SharedConfig.GetTestModelV11(ModelType.Segmentation);
            var testImage = SharedConfig.GetTestImage(ImageType.People);

            using var yolo = new Yolo(new YoloOptions
            {
                OnnxModel = model,
                Cuda = false
            });

            using var image = SKBitmap.Decode(testImage);

            // Act
            var results = yolo.RunSegmentation(image);

            // Assert
            Assert.Equal(17, results.Count);
        }
    }
}
