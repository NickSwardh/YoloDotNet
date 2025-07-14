// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Models
{
    /// <summary>
    /// Represents the result of image classification
    /// </summary>
    public class Classification : IClassification
    {
        /// <summary>
        /// Label of classified image.
        /// </summary>
        public string Label { get; set; } = default!;

        /// <summary>
        /// Confidence score of classified image.
        /// </summary>
        public double Confidence { get; set; }
    }
}
