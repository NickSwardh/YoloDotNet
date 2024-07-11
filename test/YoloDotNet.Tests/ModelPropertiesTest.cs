namespace YoloDotNet.Tests
{
    public class ModelPropertiesTest
    {
        [Theory]
        [InlineData(ModelType.Classification, 1, 3, 224, 224, 1, 0, 1000)]
        [InlineData(ModelType.ObjectDetection, 1, 3, 640, 640, 1, 8400, 84)]
        [InlineData(ModelType.ObbDetection, 1, 3, 1024, 1024, 1, 21504, 20)]
        [InlineData(ModelType.Segmentation, 1, 3, 640, 640, 1, 8400, 116, 0, 0, 1, 32, 0, 160, 160)]
        [InlineData(ModelType.PoseEstimation, 1, 3, 640, 640, 1, 8400, 56)]
        public void OnnxModel_ValidateProperties_ReturnTrue(
            ModelType modelTypeToTest,

            int inputBatch,
            int inputChannels,
            int inputWidth,
            int inputHeight,

            int output0Batch,
            int output0Channels,
            int output0Elements,
            int output0Width = 0,
            int output0Height = 0,

            int output1Batch = 0,
            int output1Channels = 0,
            int output1Elements = 0,
            int output1Width = 0,
            int output1eight = 0
            )
        {
            // Arrange
            var yolo = new Yolo(new YoloOptions
            {
                OnnxModel = SharedConfig.GetTestModelV8(modelTypeToTest),
                ModelType = modelTypeToTest,
                Cuda = false
            });

            var props = yolo.OnnxModel;

            // Act
            var input = props.Input;
            var output0 = props.Outputs[0];
            var output1 = props.Outputs[1];

            // Assert

            // Model input
            Assert.Equal(inputBatch, input.BatchSize);
            Assert.Equal(inputChannels, input.Channels);
            Assert.Equal(inputWidth, input.Width);
            Assert.Equal(inputHeight, input.Height);

            // Model output0
            Assert.Equal(output0Batch, output0.BatchSize);
            Assert.Equal(output0Channels, output0.Channels);
            Assert.Equal(output0Elements, output0.Elements);
            Assert.Equal(output0Width, output0.Width);
            Assert.Equal(output0Height, output0.Height);

            // Model output1
            Assert.Equal(output1Batch, output1.BatchSize);
            Assert.Equal(output1Channels, output1.Channels);
            Assert.Equal(output1Elements, output1.Elements);
            Assert.Equal(output1Width, output1.Width);
            Assert.Equal(output1eight, output1.Height);
        }
    }
}