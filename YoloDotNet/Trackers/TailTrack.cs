// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Trackers
{
    /// <summary>
    /// Stores a fixed-length queue of tail points to track and visualize the movement tail of an object.
    /// </summary>
    /// <param name="maxLength">Maximum number of points kept. The oldest point is dropped when full.</param>
    public class TailTrack(int maxLength)
    {
        private readonly int _maxLength = maxLength > 0
            ? maxLength
            : throw new ArgumentOutOfRangeException(nameof(maxLength), maxLength, "Must be greater than 0.");
        private readonly Queue<SKPoint> _positions = new (maxLength);

        /// <summary>
        /// Returns a copy of the current tail points as a list.
        /// </summary>
        public List<SKPoint> GetTail() => [.. _positions];

        /// <summary>
        /// Adds a new point to the tail.
        /// Automatically removes the oldest point if the tail exceeds the maximum length.
        /// </summary>
        public void AddTailPoint(SKPoint point)
        {
            if (_positions.Count >= _maxLength)
                _positions.Dequeue();

            _positions.Enqueue(point);
        }
    }
}
