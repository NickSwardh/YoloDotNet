// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Modules.Interfaces
{
    public interface ISegmentationModule : IModule
    {
        List<Segmentation> ProcessImage<T>(T image, double confidence, double pixelConfidence, double iou);
    }
}