// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

using YoloDotNet.Trackers;

namespace YoloDotNet.Tests.TrackerTests
{
    public class TrackedObjectTests
    {
        [Fact]
        public void NewTrack_HasSeededTail()
        {
            var box = SortTrackerTests.Box(100, 100, 50, 50);
            _ = new TrackedObject(box, tailLength: 10);

            Assert.NotNull(box.Tail);
            Assert.Single(box.Tail!);
            Assert.Equal(new SKPoint(125, 125), box.Tail![0]);
        }

        [Fact]
        public void Tail_EndsAtCurrentCenter()
        {
            var track = new TrackedObject(SortTrackerTests.Box(100, 100, 50, 50), tailLength: 10);

            var next = SortTrackerTests.Box(110, 100, 50, 50);
            track.TrackBoundingBox(next);

            Assert.Equal(2, next.Tail!.Count);
            Assert.Equal(new SKPoint(135, 125), next.Tail[^1]);
        }

        [Fact]
        public void Tail_IsCappedAtTailLength()
        {
            var track = new TrackedObject(SortTrackerTests.Box(0, 0, 50, 50), tailLength: 5);
            ObjectDetection last = null!;

            for (int i = 1; i <= 20; i++)
            {
                last = SortTrackerTests.Box(i * 10, 0, 50, 50);
                track.TrackBoundingBox(last);
            }

            Assert.Equal(5, last.Tail!.Count);
            Assert.Equal(new SKPoint(225, 25), last.Tail[^1]);
        }
    }
}
