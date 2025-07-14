// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Extensions
{
    public static class TrackerExtension
    {
        public static List<T> Track<T>(this List<T> detections, SortTracker sortTrack) where T : IDetection
        {
            sortTrack.UpdateTracker(detections);
            return detections;
        }
    }
}
