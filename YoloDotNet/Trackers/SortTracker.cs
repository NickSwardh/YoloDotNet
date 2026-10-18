// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Trackers
{
    /// <summary>
    /// Multi-object tracker based on SORT (Simple Online and Realtime Tracking), with ByteTrack-style
    /// matching of low-confidence detections. Gives each object a stable Id and a movement tail across frames.
    /// </summary>
    /// <remarks>
    /// Not thread-safe. Use one instance per video stream, and call <see cref="UpdateTracker{T}"/> once per frame, in order.
    /// </remarks>
    public partial class SortTracker
    {
        // Active tracks, keyed by an internal key. The public Id is only assigned once a track is confirmed.
        private readonly Dictionary<int, TrackedObject> _trackedObjects = [];

        private int _nextKey = 0;
        private int _nextId = 0;

        private readonly float _costThreshold;
        private readonly int _maxAge;
        private readonly int _tailLength;
        private readonly float _highConfidenceThreshold;
        private readonly int _minHits;

        // Cost given to pairs that must never match (different class, or too far apart).
        // Anything at or above the threshold is rejected after solving; keeping it close to real costs
        // instead of using a huge number keeps the solver numerically stable.
        private readonly float _blockedCost;

        /// <summary>
        /// Creates a new SORT tracker.
        /// </summary>
        /// <param name="costThreshold">
        /// Maximum matching cost for assigning a detection to a track. Lower = stricter matching.
        /// Cost = (1 - IoU) + center distance divided by the tracked box diagonal, so movement is measured relative to object size.
        /// </param>
        /// <param name="maxAge">Number of frames an unmatched track is kept before removal. Higher values survive longer occlusions.</param>
        /// <param name="tailLength">Maximum number of points kept in each track's tail.</param>
        /// <param name="highConfidenceThreshold">
        /// Enables ByteTrack-style two-stage matching when greater than 0. Detections with confidence at or above this value
        /// are matched first and may start new tracks. Detections below it are only used to keep existing tracks alive
        /// (e.g. partially occluded objects) and are removed from the list if they don't match a track.
        /// Run inference with a lower confidence than this value to benefit from it. Default 0 (disabled).
        /// </param>
        /// <param name="minHits">
        /// Number of matched frames before a new track is confirmed and receives an Id.
        /// Tentative tracks have a null Id and are discarded as soon as they miss a frame, so short-lived
        /// false or duplicate detections don't consume Ids. Default 3. Use 1 to assign Ids immediately.
        /// </param>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when a parameter is outside its valid range.</exception>
        public SortTracker(float costThreshold = 0.5f, int maxAge = 3, int tailLength = 30, float highConfidenceThreshold = 0f, int minHits = 3)
        {
            if (float.IsFinite(costThreshold) is false || costThreshold <= 0)
                throw new ArgumentOutOfRangeException(nameof(costThreshold), costThreshold, "Must be a finite value greater than 0.");

            if (float.IsFinite(highConfidenceThreshold) is false || highConfidenceThreshold < 0 || highConfidenceThreshold > 1)
                throw new ArgumentOutOfRangeException(nameof(highConfidenceThreshold), highConfidenceThreshold, "Must be between 0 and 1.");

            ArgumentOutOfRangeException.ThrowIfNegative(maxAge);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(tailLength);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(minHits);

            _costThreshold = costThreshold;
            _maxAge = maxAge;
            _tailLength = tailLength;
            _highConfidenceThreshold = highConfidenceThreshold;
            _minHits = minHits;
            _blockedCost = costThreshold + 1f;
        }

        /// <summary>
        /// Updates the tracker with the detections from the current frame.
        /// Sets <see cref="IDetection.Id"/> and <see cref="IDetection.Tail"/> on matched detections.
        /// </summary>
        /// <typeparam name="T">A detection type implementing <see cref="IDetection"/>.</typeparam>
        /// <param name="detections">
        /// Detections from the current frame. When <c>highConfidenceThreshold</c> is enabled, low-confidence
        /// detections that don't match an existing track are removed from this list.
        /// </param>
        /// <remarks>
        /// Steps per frame:
        /// <list type="number">
        /// <item>Predict where every track will be in this frame (Kalman filter).</item>
        /// <item>Match high-confidence detections to tracks.</item>
        /// <item>Match low-confidence detections to the remaining confirmed tracks, and drop the ones that don't match.</item>
        /// <item>Start new tracks for unmatched high-confidence detections.</item>
        /// <item>Age tracks and remove the ones that have been missing too long.</item>
        /// </list>
        /// </remarks>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="detections"/> is null.</exception>
        public void UpdateTracker<T>(List<T> detections) where T : IDetection
        {
            ArgumentNullException.ThrowIfNull(detections);

            // Predict every frame, including empty ones, so missing objects keep moving in time
            foreach (var trackedObject in _trackedObjects.Values)
                trackedObject.KalmanPredict();

            if (detections.Count > 0)
                MatchFrame(detections);

            RemoveOldTrackedObjects();
        }

        /// <summary>
        /// Creates a new track for every detection, without matching against existing tracks.
        /// </summary>
        public void CreateInitialTracks<T>(List<T> detections) where T : IDetection
        {
            ArgumentNullException.ThrowIfNull(detections);

            foreach (var detection in detections)
                CreateTrack(detection);
        }

        private void MatchFrame<T>(List<T> detections) where T : IDetection
        {
            var highIndices = new List<int>(detections.Count);
            var lowIndices = new List<int>();

            for (int i = 0; i < detections.Count; i++)
            {
                if (detections[i].Confidence >= _highConfidenceThreshold)
                    highIndices.Add(i);
                else
                    lowIndices.Add(i);
            }

            // Tracks still waiting for a match this frame. Matched tracks are removed from this list.
            var unmatchedTracks = new List<TrackedObject>(_trackedObjects.Values);

            // Stage 1: high-confidence detections
            var matchedHigh = MatchDetections(detections, highIndices, unmatchedTracks);

            // Stage 2: low-confidence detections may only keep confirmed tracks alive
            unmatchedTracks.RemoveAll(track => track.Id is null);
            var matchedLow = MatchDetections(detections, lowIndices, unmatchedTracks);

            foreach (var index in highIndices)
            {
                if (matchedHigh.Contains(index) is false)
                    CreateTrack(detections[index]);
            }

            // Remove unmatched low-confidence detections (likely noise). Back to front, so earlier indices stay valid.
            for (int i = lowIndices.Count - 1; i >= 0; i--)
            {
                if (matchedLow.Contains(lowIndices[i]) is false)
                    detections.RemoveAt(lowIndices[i]);
            }
        }

        private void CreateTrack(IDetection detection)
        {
            var track = new TrackedObject(detection, _tailLength);

            if (_minHits == 1)
                track.Id = ++_nextId;

            detection.Id = track.Id;
            _trackedObjects[++_nextKey] = track;
        }

        /// <summary>
        /// Ages all tracks by one frame. Removes tracks that have been missing for more than <c>maxAge</c> frames,
        /// and tentative tracks that missed a single frame.
        /// </summary>
        private void RemoveOldTrackedObjects()
        {
            // Removing during enumeration is safe for Dictionary in .NET Core 3.0+
            foreach (var (key, track) in _trackedObjects)
            {
                track.Age++;

                // Age is 1 right after a match, so > 1 means the track was not matched this frame
                var isTentativeAndMissed = track.Id is null && track.Age > 1;

                if (track.Age > _maxAge || isTentativeAndMissed)
                    _trackedObjects.Remove(key);
            }
        }

        /// <summary>
        /// Matches the given detections to the given tracks with the lowest total cost.
        /// Matched tracks are updated and removed from <paramref name="tracks"/>.
        /// </summary>
        /// <returns>Indices (into <paramref name="detections"/>) of the detections that were matched.</returns>
        private HashSet<int> MatchDetections<T>(List<T> detections, List<int> detectionIndices, List<TrackedObject> tracks) where T : IDetection
        {
            var matched = new HashSet<int>();

            if (tracks.Count == 0 || detectionIndices.Count == 0)
                return matched;

            var costMatrix = CalculateCostMatrix(tracks, detections, detectionIndices);
            var assignment = LAPJV.Solve(costMatrix);
            var matchedRows = new bool[assignment.Length];

            for (int row = 0; row < assignment.Length; row++)
            {
                int column = assignment[row];

                // The solver pairs as many as it can, so pairs over the threshold must be rejected here
                if (column == -1 || costMatrix[row, column] >= _costThreshold)
                    continue;

                var detection = detections[detectionIndices[column]];
                var track = tracks[row];

                track.Age = 0;
                track.Hits++;

                if (track.Id is null && track.Hits >= _minHits)
                    track.Id = ++_nextId;

                detection.Id = track.Id;
                track.TrackBoundingBox(detection);

                matched.Add(detectionIndices[column]);
                matchedRows[row] = true;
            }

            // Remove matched tracks back to front, so earlier indices stay valid
            for (int row = matchedRows.Length - 1; row >= 0; row--)
            {
                if (matchedRows[row])
                    tracks.RemoveAt(row);
            }

            return matched;
        }

        /// <summary>
        /// Builds the cost of pairing each track (row) with each detection (column).
        /// Cost = (1 - IoU with the predicted box) + distance from the predicted center relative to the box diagonal.
        /// Pairs with different classes, or with a cost at or above the threshold, get the blocked cost.
        /// </summary>
        private float[,] CalculateCostMatrix<T>(List<TrackedObject> tracks, List<T> detections, List<int> detectionIndices) where T : IDetection
        {
            var costMatrix = new float[tracks.Count, detectionIndices.Count];

            for (int row = 0; row < tracks.Count; row++)
            {
                var track = tracks[row];
                var predictedBox = track.GetPredictedBox();
                var trackLabel = track.BoundingBox.Label.Index;

                // At least 1 to avoid dividing by zero for degenerate boxes
                var diagonal = MathF.Max(Length(predictedBox.Width, predictedBox.Height), 1f);

                for (int column = 0; column < detectionIndices.Count; column++)
                {
                    var detection = detections[detectionIndices[column]];

                    if (detection.Label.Index != trackLabel)
                    {
                        costMatrix[row, column] = _blockedCost;
                        continue;
                    }

                    var detectionBox = detection.BoundingBox;
                    var iouCost = 1f - YoloCore.CalculateIoU(predictedBox, detectionBox);
                    var distance = Length(predictedBox.MidX - detectionBox.MidX, predictedBox.MidY - detectionBox.MidY);
                    var cost = iouCost + distance / diagonal;

                    costMatrix[row, column] = cost < _costThreshold ? cost : _blockedCost;
                }
            }

            return costMatrix;
        }

        private static float Length(float dx, float dy)
            => MathF.Sqrt(dx * dx + dy * dy);
    }
}
