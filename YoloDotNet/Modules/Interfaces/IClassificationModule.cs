// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Modules.Interfaces
{
    public interface IClassificationModule : IModule
    {
        List<Classification> ProcessImage<T>(T image, double classes, double pixelConfidence, double iou);
    }
}