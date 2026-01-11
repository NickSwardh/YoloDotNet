// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Enums
{
    /// <summary>
    /// Specifies the placement of labels relative to bounding boxes.
    /// </summary>
    public enum LabelPosition
    {
        /// <summary>
        /// Automatically determines the best label position based on available space.
        /// Tries positions in priority order: top-left, bottom-left, inside top-left, inside bottom-left.
        /// Labels will always remain within image bounds.
        /// </summary>
        Auto,

        /// <summary>
        /// Places labels at the top-left corner of the bounding box.
        /// If there is insufficient space, the label will be placed inside the box at the top-left corner.
        /// </summary>
        TopLeft,

        /// <summary>
        /// Places labels at the top-right corner of the bounding box.
        /// If there is insufficient space, the label will be placed inside the box at the top-right corner.
        /// </summary>
        TopRight,

        /// <summary>
        /// Places labels at the bottom-left corner of the bounding box.
        /// If there is insufficient space, the label will be placed inside the box at the bottom-left corner.
        /// </summary>
        BottomLeft,

        /// <summary>
        /// Places labels at the bottom-right corner of the bounding box.
        /// If there is insufficient space, the label will be placed inside the box at the bottom-right corner.
        /// </summary>
        BottomRight
    }
}
