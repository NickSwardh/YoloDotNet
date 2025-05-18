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
                var model = SharedConfig.GetTestModelV8(ModelType.ObjectDetection);
                var mock = new Yolo(new YoloOptions
                {
                    OnnxModel = model,
                    Cuda = useCuda
                });
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
