// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Tests.ObjectDetectionTests
{
    public class Yolov12ObjectDetectionTests
    {
        [Fact]
        public void RunObjectDetection_Yolov12_GetExpectedNumberOfObjects_AssertTrue()
        {
            // Arrange
            var model = SharedConfig.GetTestModelV12(ModelType.ObjectDetection);
            var testImage = SharedConfig.GetTestImage(ImageType.Street);

            using var yolo = new Yolo(new YoloOptions
            {
                OnnxModel = model
            });

            using var image = SKBitmap.Decode(testImage);

            // Act
            var results = yolo.RunObjectDetection(image, 0.23, 0.7);

            // Assert
            Assert.Equal(35, results.Count);
        }
    }
}
