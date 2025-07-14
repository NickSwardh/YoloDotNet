// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Tests.PoseEstimationTests
{
    public class Yolov8PoseEstimationTests
    {
        [Fact]
        public void RunPoseEstimation_Yolov8_GetExpectedNumberOfPoseEstimations_AssertTrue()
        {
            // Arrange
            var model = SharedConfig.GetTestModelV8(ModelType.PoseEstimation);
            var testImage = SharedConfig.GetTestImage(ImageType.Crosswalk);

            using var yolo = new Yolo(new YoloOptions
            {
                OnnxModel = model,
                Cuda = false,
                ImageResize = ImageResize.Proportional
            });

            using var image = SKBitmap.Decode(testImage);

            // Act
            var results = yolo.RunPoseEstimation(image, 0.23, 0.7);

            // Assert
            Assert.Equal(11, results.Count);
        }
    }
}
