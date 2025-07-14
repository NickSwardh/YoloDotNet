// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Models.Interfaces
{
    public interface IClassification
    {
        string Label { get; set; }
        double Confidence { get; set; }
    }
}
