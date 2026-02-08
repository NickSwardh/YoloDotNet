// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Tests.SegmentationTests
{
    public class Yolov8SegmentationTests
    {
        [Fact]
        public void RunSegmentation_Yolov8_GetExpectedNumberOfSegmentations_AssertTrue()
        {
            // Arrange
            var model = SharedConfig.GetTestModelV8(ModelType.Segmentation);
            var testImage = SharedConfig.GetTestImage(ImageType.Segmentation);

            using var yolo = new Yolo(new YoloOptions
            {
                ExecutionProvider = new CpuExecutionProvider(model),
            });

            using var image = SKBitmap.Decode(testImage);

            // Act
            var results = yolo.RunSegmentation(image, 0.23, 0.7);

            // Assert
            Assert.Equal(21, results.Count);
        }

        [Fact]
        public void RunSegmentation_Yolov8_ROI_GetExpectedNumberOfSegmentations_AssertTrue()
        {
            // Arrange
            var model = SharedConfig.GetTestModelV8(ModelType.Segmentation);
            var testImage = SharedConfig.GetTestImage(ImageType.Segmentation);

            using var yolo = new Yolo(new YoloOptions
            {
                ExecutionProvider = new CpuExecutionProvider(model),
            });

            using var image = SKBitmap.Decode(testImage);

            var roi = SKRectI.Create(228, 415, 163, 324);

            // Act
            var results = yolo.RunSegmentation(image, 0.23, 0.7, roi: roi);

            // Assert
            Assert.Single(results);
        }
    }
}
