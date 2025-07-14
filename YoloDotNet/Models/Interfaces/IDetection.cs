﻿// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Models.Interfaces
{
    public interface IDetection
    {
        LabelModel Label { get; init; }
        double Confidence { get; init; }
        SKRectI BoundingBox { get; init; }

        // Optional properties used for SORT Tracking
        int? Id { get; set; }

        List<SKPoint>? Tail { get; set; }
    }
}