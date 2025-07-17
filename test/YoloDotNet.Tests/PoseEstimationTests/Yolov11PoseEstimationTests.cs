// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Tests.PoseEstimationTests
{
    public class Yolov11PoseEstimationTests
    {
        [Fact]
        public void RunPoseEstimation_Yolov11_GetExpectedNumberOfPoseEstimations_AssertTrue()
        {
            // Arrange
            var model = SharedConfig.GetTestModelV11(ModelType.PoseEstimation);
            var testImage = SharedConfig.GetTestImage(ImageType.Crosswalk);

            using var yolo = new Yolo(new YoloOptions
            {
                OnnxModel = model
            });

            using var image = SKBitmap.Decode(testImage);

            // Act
            var results = yolo.RunPoseEstimation(image, 0.23, 0.7);

            // Assert
            Assert.Equal(11, results.Count);
        }
    }
}
