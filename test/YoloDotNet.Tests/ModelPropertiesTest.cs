// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Tests
{
    public class ModelPropertiesTest
    {
        // Shapes from Ultralytics ONNX v8 models
        [Theory]
        [InlineData(ModelType.Classification, 1, 3, 224, 224, 1, 1000)]
        [InlineData(ModelType.ObjectDetection, 1, 3, 640, 640, 1, 84, 8400)]
        [InlineData(ModelType.ObbDetection, 1, 3, 1024, 1024, 1, 20, 21504)]
        [InlineData(ModelType.Segmentation, 1, 3, 640, 640, 1, 116, 8400, 1, 32, 160, 160)]
        [InlineData(ModelType.PoseEstimation, 1, 3, 640, 640, 1, 56, 8400)]
        public void OnnxModel_ValidateProperties_ReturnTrue(
            ModelType modelTypeToTest,

            int inputBatch,
            int inputChannels,
            int inputHeight,
            int inputWidth,

            int output0Batch,
            int output0Channels,
            int output0Elements = 0,

            int output1Batch = 0,
            int output1Channels = 0,
            int output1Height = 0,
            int output1Width = 0
            )
        {
            // Arrange
            using var yolo = new Yolo(new YoloOptions
            {
                ExecutionProvider = new CpuExecutionProvider(SharedConfig.GetTestModelV8(modelTypeToTest))
            });

            var props = yolo.OnnxModel;

            // Act
            var input = props.InputShapes.First().Value;
            var output0 = props.OutputShapes.ElementAt(0).Value;
            //var output1 = props.OutputShapes.ElementAt(1).Value;

            // Assert

            // Model input
            Assert.Equal(inputBatch, input[0]);
            Assert.Equal(inputChannels, input[1]);
            Assert.Equal(inputHeight, input[2]);
            Assert.Equal(inputWidth, input[3]);

            // Model output0
            Assert.Equal(output0Batch, output0[0]);
            Assert.Equal(output0Channels, output0[1]);

            // Skip third dimension check for classification models.
            if (props.ModelType is not ModelType.Classification)
                Assert.Equal(output0Elements, output0[2]);

            // If there is a second output (eg., for segmentation), validate that as well
            if (props.OutputShapes.Count > 1)
            {
                var output1 = props.OutputShapes.ElementAt(1).Value;

                // Model output1
                Assert.Equal(output1Batch, output1[0]);
                Assert.Equal(output1Channels, output1[1]);
                Assert.Equal(output1Height, output1[2]);
                Assert.Equal(output1Width, output1[3]);
            }
        }
    }
}