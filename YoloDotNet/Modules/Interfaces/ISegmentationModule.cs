// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Modules.Interfaces
{
    public interface ISegmentationModule : IModule
    {
        List<Segmentation> ProcessImage<T>(T image, double confidence, double pixelConfidence, double iou, SKRectI? roi = null);
    }
}