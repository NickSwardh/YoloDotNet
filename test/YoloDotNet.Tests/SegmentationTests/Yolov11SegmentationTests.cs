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

            var yolo = new Yolo(new YoloOptions
            {
                OnnxModel = model,
                ModelType = ModelType.Segmentation,
                HwAccelerator = HwAcceleratorType.None
            });

            var image = SKImage.FromEncodedData(testImage);

            // Act
            var results = yolo.RunSegmentation(image, 0.25, 0.65, 0.45);

            // Assert
            Assert.Equal(17, results.Count);
        }
    }
}
