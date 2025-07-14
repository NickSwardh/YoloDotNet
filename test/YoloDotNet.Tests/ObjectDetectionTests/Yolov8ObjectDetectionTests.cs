// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Tests.ObjectDetectionTests
{
    public class Yolov8ObjectDetectionTests
    {
        [Fact]
        public void RunObjectDetection_Yolov8_GetExpectedNumberOfObjects_AssertTrue()
        {
            // Arrange
            var model = SharedConfig.GetTestModelV8(ModelType.ObjectDetection);
            var testImage = SharedConfig.GetTestImage(ImageType.Street);

            using var yolo = new Yolo(new YoloOptions
            {
                OnnxModel = model,
                Cuda = false,
                ImageResize = ImageResize.Proportional
            });

            using var image = SKBitmap.Decode(testImage);

            // Act
            var results = yolo.RunObjectDetection(image, 0.23, 0.7);

            // Assert
            Assert.Equal(33, results.Count);
        }
    }
}
