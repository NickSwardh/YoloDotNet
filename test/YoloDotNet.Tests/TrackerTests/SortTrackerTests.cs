// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

using YoloDotNet.Trackers;

namespace YoloDotNet.Tests.TrackerTests
{
    public class SortTrackerTests
    {
        internal static ObjectDetection Box(int x, int y, int w, int h)
            => new()
            {
                Label = new LabelModel { Index = 0, Name = "person" },
                Confidence = 0.9,
                BoundingBox = new SKRectI(x, y, x + w, y + h)
            };

        [Fact]
        public void LargeFastMovingBox_KeepsSameId()
        {
            var tracker = new SortTracker(costThreshold: 0.5f, maxAge: 5, tailLength: 30, minHits: 1);
            int? firstId = null;

            for (int frame = 0; frame < 30; frame++)
            {
                var detections = new List<ObjectDetection> { Box(50 + frame * 40, 60, 400, 600) };
                tracker.UpdateTracker(detections);

                firstId ??= detections[0].Id;
                Assert.Equal(firstId, detections[0].Id);
            }
        }

        [Fact]
        public void SmallBoxJumpingFar_GetsNewId()
        {
            var tracker = new SortTracker(minHits: 1);

            var first = new List<ObjectDetection> { Box(100, 100, 30, 30) };
            tracker.UpdateTracker(first);

            var second = new List<ObjectDetection> { Box(300, 100, 30, 30) };
            tracker.UpdateTracker(second);

            Assert.NotEqual(first[0].Id, second[0].Id);
        }

        [Fact]
        public void NearbySmallBoxesMovingInParallel_KeepOwnIds()
        {
            var tracker = new SortTracker(costThreshold: 0.5f, maxAge: 5, tailLength: 30, minHits: 1);
            int? idA = null, idB = null;

            for (int frame = 0; frame < 30; frame++)
            {
                var a = Box(100 + frame * 5, 100, 30, 30);
                var b = Box(100 + frame * 5, 140, 30, 30);
                tracker.UpdateTracker(new List<ObjectDetection> { a, b });

                idA ??= a.Id;
                idB ??= b.Id;

                Assert.Equal(idA, a.Id);
                Assert.Equal(idB, b.Id);
            }

            Assert.NotEqual(idA, idB);
        }

        [Fact]
        public void TrackMissingWithinMaxAge_IsRematched()
        {
            var tracker = new SortTracker(costThreshold: 0.5f, maxAge: 5, tailLength: 30, minHits: 1);

            var first = new List<ObjectDetection> { Box(100, 100, 100, 200) };
            tracker.UpdateTracker(first);

            for (int i = 0; i < 3; i++)
                tracker.UpdateTracker(new List<ObjectDetection>());

            var again = new List<ObjectDetection> { Box(100, 100, 100, 200) };
            tracker.UpdateTracker(again);

            Assert.Equal(first[0].Id, again[0].Id);
        }

        [Fact]
        public void TrackMissingLongerThanMaxAge_GetsNewId()
        {
            var tracker = new SortTracker(costThreshold: 0.5f, maxAge: 3, tailLength: 30, minHits: 1);

            var first = new List<ObjectDetection> { Box(100, 100, 100, 200) };
            tracker.UpdateTracker(first);

            for (int i = 0; i < 6; i++)
                tracker.UpdateTracker(new List<ObjectDetection>());

            var again = new List<ObjectDetection> { Box(100, 100, 100, 200) };
            tracker.UpdateTracker(again);

            Assert.NotEqual(first[0].Id, again[0].Id);
        }

        [Fact]
        public void MovingBoxMissedForFewFrames_IsRematchedAtPredictedPosition()
        {
            var tracker = new SortTracker(costThreshold: 0.5f, maxAge: 10, tailLength: 30, minHits: 1);
            int? id = null;

            // Person (60x160) walking 15 px/frame
            for (int frame = 0; frame < 15; frame++)
            {
                var d = new List<ObjectDetection> { Box(100 + frame * 15, 200, 60, 160) };
                tracker.UpdateTracker(d);
                id ??= d[0].Id;
            }

            // Occluded for 4 frames while still walking
            for (int frame = 15; frame < 19; frame++)
                tracker.UpdateTracker(new List<ObjectDetection>());

            var again = new List<ObjectDetection> { Box(100 + 19 * 15, 200, 60, 160) };
            tracker.UpdateTracker(again);

            Assert.Equal(id, again[0].Id);
        }

        [Fact]
        public void DifferentClass_IsNotMatched()
        {
            var tracker = new SortTracker(minHits: 1);

            var first = new List<ObjectDetection> { Box(100, 100, 100, 200) };
            tracker.UpdateTracker(first);

            var other = new ObjectDetection
            {
                Label = new LabelModel { Index = 1, Name = "dog" },
                Confidence = 0.9,
                BoundingBox = new SKRectI(100, 100, 200, 300)
            };
            var second = new List<ObjectDetection> { other };
            tracker.UpdateTracker(second);

            Assert.NotEqual(first[0].Id, other.Id);
        }

        [Fact]
        public void LowConfidenceDetection_KeepsTrackAlive_AndUnmatchedLowIsDropped()
        {
            var tracker = new SortTracker(costThreshold: 0.5f, maxAge: 5, tailLength: 30, highConfidenceThreshold: 0.5f, minHits: 1);

            var first = new List<ObjectDetection> { Box(100, 100, 100, 200) };
            tracker.UpdateTracker(first);

            var weakMatch = new ObjectDetection { Label = new LabelModel { Name = "person" }, Confidence = 0.2, BoundingBox = new SKRectI(105, 100, 205, 300) };
            var weakNoise = new ObjectDetection { Label = new LabelModel { Name = "person" }, Confidence = 0.2, BoundingBox = new SKRectI(800, 100, 900, 300) };
            var frame = new List<ObjectDetection> { weakMatch, weakNoise };
            tracker.UpdateTracker(frame);

            Assert.Equal(first[0].Id, weakMatch.Id);
            Assert.Single(frame);
            Assert.Same(weakMatch, frame[0]);
        }

        [Fact]
        public void NewTrack_GetsIdOnlyAfterMinHits()
        {
            var tracker = new SortTracker(minHits: 3);
            var ids = new List<int?>();

            for (int frame = 0; frame < 4; frame++)
            {
                var box = Box(100 + frame * 5, 100, 50, 100);
                tracker.UpdateTracker(new List<ObjectDetection> { box });
                ids.Add(box.Id);
            }

            Assert.Null(ids[0]);
            Assert.Null(ids[1]);
            Assert.Equal(1, ids[2]);
            Assert.Equal(1, ids[3]);
        }

        [Fact]
        public void FlickeringDetections_DoNotConsumeIds()
        {
            var tracker = new SortTracker(minHits: 3);

            // A false detection appearing every other frame never confirms
            for (int frame = 0; frame < 10; frame++)
            {
                var detections = frame % 2 == 0
                    ? new List<ObjectDetection> { Box(300, 300, 40, 80) }
                    : new List<ObjectDetection>();
                tracker.UpdateTracker(detections);
            }

            var real = Box(600, 100, 50, 100);
            for (int frame = 0; frame < 3; frame++)
            {
                real = Box(600, 100, 50, 100);
                tracker.UpdateTracker(new List<ObjectDetection> { real });
            }

            Assert.Equal(1, real.Id);
        }

        [Fact]
        public void Crowd_StationaryPeople_KeepIds()
        {
            var tracker = new SortTracker(costThreshold: 0.7f, maxAge: 30, tailLength: 30, minHits: 1);
            var rnd = new Random(42);
            var positions = Enumerable.Range(0, 30).Select(i => (x: (i % 10) * 90 + 10, y: (i / 10) * 150 + 20)).ToList();
            Dictionary<int, int?> firstIds = [];

            for (int frame = 0; frame < 50; frame++)
            {
                // Shuffled order and a little jitter, like a real detector
                var detections = positions
                    .Select((p, i) => (i, box: Box(p.x + rnd.Next(-2, 3), p.y + rnd.Next(-2, 3), 30, 70)))
                    .OrderBy(_ => rnd.Next())
                    .ToList();

                tracker.UpdateTracker(detections.Select(d => d.box).ToList());

                foreach (var (i, box) in detections)
                {
                    if (firstIds.TryGetValue(i, out var id))
                        Assert.Equal(id, box.Id);
                    else
                        firstIds[i] = box.Id;
                }
            }

            Assert.Equal(30, firstIds.Values.Distinct().Count());
        }

        [Theory]
        [InlineData(0f, 3, 30, 0f, 3)]
        [InlineData(float.NaN, 3, 30, 0f, 3)]
        [InlineData(0.5f, -1, 30, 0f, 3)]
        [InlineData(0.5f, 3, 0, 0f, 3)]
        [InlineData(0.5f, 3, 30, 1.5f, 3)]
        [InlineData(0.5f, 3, 30, 0f, 0)]
        public void InvalidParameters_Throw(float costThreshold, int maxAge, int tailLength, float highConfidenceThreshold, int minHits)
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new SortTracker(costThreshold, maxAge, tailLength, highConfidenceThreshold, minHits));
        }

        [Fact]
        public void NullDetections_Throws()
        {
            Assert.Throws<ArgumentNullException>(() => new SortTracker().UpdateTracker<ObjectDetection>(null!));
        }

        [Fact]
        public void TrackedBox_TailEndsAtCurrentCenterAndGrows()
        {
            var tracker = new SortTracker(costThreshold: 0.5f, maxAge: 5, tailLength: 10, minHits: 1);

            for (int frame = 0; frame < 5; frame++)
            {
                var box = Box(100 + frame * 10, 100, 100, 200);
                tracker.UpdateTracker(new List<ObjectDetection> { box });

                Assert.NotNull(box.Tail);
                Assert.Equal(frame + 1, box.Tail!.Count);
                Assert.Equal(new SKPoint(box.BoundingBox.MidX, box.BoundingBox.MidY), box.Tail[^1]);
            }
        }
    }
}

