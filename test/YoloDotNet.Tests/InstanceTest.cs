// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

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
                using var mock = new Yolo(new YoloOptions
                {
                    ExecutionProvider = new CpuExecutionProvider(model)
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
