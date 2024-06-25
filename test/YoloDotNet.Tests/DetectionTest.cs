namespace YoloDotNet.Tests
{
    public class DetectionTest
    {
        [Fact]
        public void RunClassification_LabelImageCorrectly_AssertTrue()
        {
            // Arrange
            var model  = SharedConfig.GetTestModel(ModelType.Classification);
            var testImage = SharedConfig.GetTestImage(ImageType.Hummingbird);

            var yolo = new Yolo(model, false);
            var image = Image.Load<Rgba32>(testImage);

            // Act
            var classification = yolo.RunClassification(image, 1);

            // Assert
            Assert.Equal("hummingbird", classification[0].Label);
        }

        [Fact]
        public void RunObjectDetection_GetExpectedNumberOfObjects_AssertTrue()
        {
            // Arrange
            var model = SharedConfig.GetTestModel(ModelType.ObjectDetection);
            var testImage = SharedConfig.GetTestImage(ImageType.Street);

            var yolo = new Yolo(model, false);
            var image = Image.Load<Rgba32>(testImage);

            // Act
            var results = yolo.RunObjectDetection(image, 0.25, 0.45);

            // Assert
            Assert.Equal(31, results.Count);
        }

        [Fact]
        public void RunObbDetection_GetExpectedNumberOfObjects_AssertTrue()
        {
            // Arrange
            var model = SharedConfig.GetTestModel(ModelType.ObbDetection);
            var testImage = SharedConfig.GetTestImage(ImageType.Island);

            var yolo = new Yolo(model, false);
            var image = Image.Load<Rgba32>(testImage);

            // Act
            var results = yolo.RunObbDetection(image, 0.25, 0.45);

            // Assert
            Assert.Equal(5, results.Count);
        }

        [Fact]
        public void RunSegmentation_GetExpectedNumberOfSegmentations_AssertTrue()
        {
            // Arrange
            var model = SharedConfig.GetTestModel(ModelType.Segmentation);
            var testImage = SharedConfig.GetTestImage(ImageType.People);

            var yolo = new Yolo(model, false);
            var image = Image.Load<Rgba32>(testImage);

            // Act
            var results = yolo.RunSegmentation(image, 0.25, 0.45);

            // Assert
            Assert.Equal(20, results.Count);
        }

        [Fact]
        public void RunPoseEstimation_GetExpectedNumberOfPoseEstimations_AssertTrue()
        {
            // Arrange
            var model = SharedConfig.GetTestModel(ModelType.PoseEstimation);
            var testImage = SharedConfig.GetTestImage(ImageType.Crosswalk);

            var yolo = new Yolo(model, false);
            var image = Image.Load<Rgba32>(testImage);

            // Act
            var results = yolo.RunPoseEstimation(image, 0.25, 0.45);

            // Assert
            Assert.Equal(10, results.Count);
        }
    }
}
