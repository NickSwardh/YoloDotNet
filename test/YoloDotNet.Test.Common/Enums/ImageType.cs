// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Test.Common.Enums
{
    /// <summary>
    /// Image types used by tests/benchmarks. Primary values are task-based names.
    /// Backwards-compatible aliases map original image names to the corresponding task.
    /// </summary>
    public enum ImageType
    {
        Classification,
        ObjectDetection,
        PoseEstimation,
        Segmentation,
        ObbDetection,
    }
}
