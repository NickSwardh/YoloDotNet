// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Tests.OBBDetectionTests
{
    public class Yolov26OBBDetectionTests
    {
        [Fact]
        public void RunObbDetection_Yolov26_GetExpectedNumberOfObjects_AssertTrue()
        {
            // Arrange
            var model = SharedConfig.GetTestModelV26(ModelType.ObbDetection);
            var testImage = SharedConfig.GetTestImage(ImageType.ObbDetection);

            using var yolo = new Yolo(new YoloOptions
            {
                ExecutionProvider = new CpuExecutionProvider(model),
                ImageResize = ImageResize.Proportional // Explicitly set to proportional resizing for clarity of the test.
            });

            //var image = SKImage.FromEncodedData(testImage);
            using var image = SKBitmap.Decode(testImage);

            // Act
            var results = yolo.RunObbDetection(image, 0.25, 0.45);

            // Assert
            Assert.Equal(5, results.Count);
        }

        [Fact]
        public void RunObbDetection_Yolov26_EnforceProportionalPreProcessing_AssertTrue()
        {
            // Arrange
            var model = SharedConfig.GetTestModelV26(ModelType.ObbDetection);
            var testImage = SharedConfig.GetTestImage(ImageType.ObbDetection);

            using var yolo = new Yolo(new YoloOptions
            {
                ExecutionProvider = new CpuExecutionProvider(model),
                ImageResize = ImageResize.Stretched // This will be overridden to Proportional internally for OBB detection.
            });

            //var image = SKImage.FromEncodedData(testImage);
            using var image = SKBitmap.Decode(testImage);

            var roi = SKRectI.Create(800, 6, 211, 193);

            // Act
            var results = yolo.RunObbDetection(image, 0.25, 0.45, roi);

            // Assert
            Assert.Single(results);
        }
    }
}
