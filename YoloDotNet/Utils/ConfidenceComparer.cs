// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Utils
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
