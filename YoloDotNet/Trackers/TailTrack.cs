// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Trackers
{
    /// <summary>
    /// Stores a fixed-length queue of tail points to track and visualize the movement tail of an object.
    /// </summary>
    /// <param name="maxLength"></param>
    public class TailTrack(int maxLength)
    {
        private readonly int _maxLength = maxLength;
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
