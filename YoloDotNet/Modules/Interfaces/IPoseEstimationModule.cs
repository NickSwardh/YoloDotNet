// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Modules.Interfaces
{
    internal interface IPoseEstimationModule : IModule
    {
        List<PoseEstimation> ProcessImage<T>(T image, double confidence, double pixelConfidence, double iou);
    }
}