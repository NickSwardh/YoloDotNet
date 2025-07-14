// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Tests.ClassificationTests
{
    public class Yolov11ClassificationTests
    {
        [Fact]
        public void RunClassification_Yolov11_LabelImageCorrectly_AssertTrue()
        {
            // Arrange
            var model = SharedConfig.GetTestModelV11(ModelType.Classification);
            var testImage = SharedConfig.GetTestImage(ImageType.Hummingbird);

            using var yolo = new Yolo(new YoloOptions
            {
                OnnxModel = model,
                Cuda = false
            });

            using var image = SKBitmap.Decode(testImage);

            // Act
            var classification = yolo.RunClassification(image, 1);

            // Assert
            Assert.Equal("hummingbird", classification[0].Label);
        }
    }
}
