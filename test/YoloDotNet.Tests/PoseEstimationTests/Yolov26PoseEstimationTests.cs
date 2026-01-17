// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Tests.PoseEstimationTests
{
    public class Yolov26PoseEstimationTests
    {
        [Fact]
        public void RunPoseEstimation_Yolov26_GetExpectedNumberOfPoseEstimations_AssertTrue()
        {
            // Arrange
            var model = SharedConfig.GetTestModelV26(ModelType.PoseEstimation);
            var testImage = SharedConfig.GetTestImage(ImageType.PoseEstimation);

            using var yolo = new Yolo(new YoloOptions
            {
                ExecutionProvider = new CpuExecutionProvider(model)
            });

            using var image = SKBitmap.Decode(testImage);

            // Act
            var results = yolo.RunPoseEstimation(image, 0.23, 0.7);

            // Assert
            Assert.Equal(9, results.Count);
        }
    }
}
