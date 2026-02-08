// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Tests.ObjectDetectionTests
{
    public class Yolov26ObjectDetectionTests
    {
        [Fact]
        public void RunObjectDetection_Yolov26_GetExpectedNumberOfObjects_AssertTrue()
        {
            // Arrange
            var model = SharedConfig.GetTestModelV26(ModelType.ObjectDetection);
            var testImage = SharedConfig.GetTestImage(ImageType.ObjectDetection);

            using var yolo = new Yolo(new YoloOptions
            {
                ExecutionProvider = new CpuExecutionProvider(model)
            });

            using var image = SKBitmap.Decode(testImage);

            // Act
            var results = yolo.RunObjectDetection(image, 0.23, 0.7);

            // Assert
            Assert.Equal(29, results.Count);
        }

        [Fact]
        public void RunObjectDetection_Yolov26_ROI_GetExpectedNumberOfObjects_AssertTrue()
        {
            // Arrange
            var model = SharedConfig.GetTestModelV26(ModelType.ObjectDetection);
            var testImage = SharedConfig.GetTestImage(ImageType.ObjectDetection);

            using var yolo = new Yolo(new YoloOptions
            {
                ExecutionProvider = new CpuExecutionProvider(model)
            });

            using var image = SKBitmap.Decode(testImage);

            var roi = SKRectI.Create(246, 350, 265, 117);

            // Act
            var results = yolo.RunObjectDetection(image, 0.23, 0.7, roi);

            // Assert
            Assert.Equal(2, results.Count);
        }
    }
}
