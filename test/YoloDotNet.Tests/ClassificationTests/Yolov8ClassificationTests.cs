// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Tests.ClassificationTests
{
    public class Yolov8ClassificationTests
    {
        [Fact]
        public void RunClassification_Yolov8_LabelImageCorrectly_AssertTrue()
        {
            // Arrange
            var model = SharedConfig.GetTestModelV8(ModelType.Classification);
            var testImage = SharedConfig.GetTestImage(ImageType.Classification);

            using var yolo = new Yolo(new YoloOptions
            {
                 ExecutionProvider = new CpuExecutionProvider(model)
            });

            using var image = SKBitmap.Decode(testImage);

            // Act
            var classification = yolo.RunClassification(image, 1);

            // Assert
            Assert.Equal("hummingbird", classification[0].Label);
        }
    }
}
