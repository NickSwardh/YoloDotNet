namespace YoloDotNet.Tests
{
    public class InstanceTest
    {
        [Theory]
        [InlineData(true)]
        [InlineData(false)]
        public void InstantiateYoloDotNet_WithCpuOrCuda_ReturnTrue(bool useCuda)
        {
            // Arrange
            var result = true;

            // Act
            try
            {
                var mock = new Yolo(Config.GetTestModel(ModelType.ObjectDetection), useCuda);
            }
            catch (Exception)
            {
                result = false;
            }

            // Assert
            Assert.True(result);
        }
    }
}
