namespace YoloDotNet.Tests.ClassificationTests
{
    public class Yolov8ClassificationTests
    {
        [Fact]
        public void RunClassification_Yolov8_LabelImageCorrectly_AssertTrue()
        {
            // Arrange
            var model = SharedConfig.GetTestModelV8(ModelType.Classification);
            var testImage = SharedConfig.GetTestImage(ImageType.Hummingbird);

            var yolo = new Yolo(new YoloOptions
            {
                OnnxModel = model,
                ModelType = ModelType.Classification,
                Cuda = false
            });

            var image = SKImage.FromEncodedData(testImage);

            // Act
            var classification = yolo.RunClassification(image, 1);

            // Assert
            Assert.Equal("hummingbird", classification[0].Label);
        }
    }
}
