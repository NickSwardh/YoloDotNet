// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Handlers
{
    internal class ConfidenceComparer : IComparer<ObjectResult>
    {
        public static readonly ConfidenceComparer Instance = new();

        public int Compare(ObjectResult? x, ObjectResult? y)
        {
            if (x is null) return 1;  // Nulls go last in descending order
            if (y is null) return -1;

            if (x.Confidence > y.Confidence) return -1;
            if (x.Confidence < y.Confidence) return 1;

            return 0;
        }
    }
}
