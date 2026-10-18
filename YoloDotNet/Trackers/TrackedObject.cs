// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Trackers
{
    /// <summary>
    /// A single object followed by <see cref="SortTracker"/>: its last detection, motion estimate and tail.
    /// </summary>
    public class TrackedObject
    {
        private readonly TailTrack _tailTracker;

        /// <summary>
        /// Creates a new track from a detection and gives the detection its first tail point.
        /// </summary>
        public TrackedObject(IDetection boundingBox, int tailLength = 30)
        {
            ArgumentNullException.ThrowIfNull(boundingBox);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(tailLength);

            BoundingBox = boundingBox;
            Kalman = new KalmanFilter(boundingBox.BoundingBox.MidX, boundingBox.BoundingBox.MidY);

            _tailTracker = new TailTrack(tailLength);
            AddTailPoint(boundingBox);
        }

        /// <summary>
        /// The most recent detection matched to this track.
        /// </summary>
        public IDetection BoundingBox { get; private set; }

        /// <summary>
        /// Frames since this track was last matched. Reset to 0 on every match.
        /// </summary>
        public int Age { get; set; } = 0;

        /// <summary>
        /// Number of frames this track has been matched to a detection, including the first.
        /// </summary>
        public int Hits { get; set; } = 1;

        /// <summary>
        /// Public track Id. Null while the track is tentative (not yet confirmed).
        /// </summary>
        public int? Id { get; set; }

        /// <summary>
        /// Motion model used to predict where the object will be in the next frame.
        /// </summary>
        public KalmanFilter Kalman { get; }

        /// <summary>
        /// Updates the track with a newly matched detection: stores it, extends the tail and corrects the motion estimate.
        /// </summary>
        public void TrackBoundingBox(IDetection detection)
        {
            ArgumentNullException.ThrowIfNull(detection);

            BoundingBox = detection;
            AddTailPoint(detection);

            var box = detection.BoundingBox;
            Kalman.Update(box.MidX, box.MidY);
        }

        /// <summary>
        /// Moves the motion estimate one frame ahead. Call once per frame.
        /// </summary>
        public void KalmanPredict()
            => Kalman.Predict();

        /// <summary>
        /// Returns the predicted state as [x, y, dx, dy]. See <see cref="KalmanFilter.GetState"/>.
        /// </summary>
        public float[] GetPredictedState()
            => Kalman.GetState();

        /// <summary>
        /// Returns the last seen bounding box, moved so its center is at the predicted position.
        /// </summary>
        public SKRectI GetPredictedBox()
        {
            var box = BoundingBox.BoundingBox;
            var state = Kalman.GetState();

            var dx = (int)MathF.Round(state[0] - box.MidX);
            var dy = (int)MathF.Round(state[1] - box.MidY);

            return new SKRectI(box.Left + dx, box.Top + dy, box.Right + dx, box.Bottom + dy);
        }

        /// <summary>
        /// Adds the detection's center to the tail, then gives the detection a copy of the tail.
        /// The point is added first so the drawn tail ends at the current box.
        /// </summary>
        private void AddTailPoint(IDetection detection)
        {
            var box = detection.BoundingBox;
            _tailTracker.AddTailPoint(new SKPoint(box.MidX, box.MidY));

            // Each detection gets its own copy, since the tail keeps changing after this frame
            detection.Tail = _tailTracker.GetTail();
        }
    }
}
