// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Tests.OBBDetectionTests
{
    public class Yolov8OBBDetectionTests
    {
        [Fact]
        public void RunObbDetection_Yolov8_GetExpectedNumberOfObjects_AssertTrue()
        {
            // Arrange
            var model = SharedConfig.GetTestModelV8(ModelType.ObbDetection);
            var testImage = SharedConfig.GetTestImage(ImageType.Island);

            using var yolo = new Yolo(new YoloOptions
            {
                OnnxModel = model
            });

            //var image = SKImage.FromEncodedData(testImage);
            using var image = SKBitmap.Decode(testImage);

            // Act
            var results = yolo.RunObbDetection(image, 0.25, 0.45);

            // Assert
            Assert.Equal(5, results.Count);
        }
    }
}
