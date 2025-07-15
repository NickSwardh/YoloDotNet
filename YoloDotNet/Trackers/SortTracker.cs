// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Trackers
{
    public partial class SortTracker
    {
        private readonly Dictionary<int, TrackedObject> _trackedObjects = [];

        private int _nextId = 0;
        private readonly int _maxAge;
        private readonly float _costThreshold;
        private readonly int _tailLength;

        public SortTracker(float costThreshold = 0.5f, int maxAge = 3, int tailLength = 30)
        {
            _costThreshold = costThreshold;
            _maxAge = maxAge;
            _tailLength = tailLength;
        }

        /// <summary>
        /// Updates the tracker state with the current detections.
        /// </summary>
        /// <typeparam name="T">A detection type implementing <see cref="IDetection"/>.</typeparam>
        /// <param name="detections">List of detections from the current frame.</param>
        /// <remarks>
        /// The method performs the following steps:
        /// <list type="bullet">
        /// <item>Predicts new positions of existing tracked objects using a Kalman filter.</item>
        /// <item>Matches current detections to active tracks using a cost matrix and assignment algorithm.</item>
        /// <item>Adds new detections as untracked objects when no matching track is found.</item>
        /// <item>Initializes tracking if no active tracks exist.</item>
        /// <item>Removes old tracked objects that have exceeded the maximum allowed age.</item>
        /// </list>
        /// </remarks>
        public void UpdateTracker<T>(List<T> detections) where T : IDetection
        {
            // If there is nothing to track, no further processing needed...
            if (detections.Count == 0)
            {
                RemoveOldTrackedObjects();
                return;
            }

            // Predict new positions using Kalman Filter
            foreach (var trackedObject in _trackedObjects.Values)
                trackedObject.KalmanPredict();

            // Get previous boundingboxes based on previous _trackCounter
            var activeTracks = _trackedObjects.Where(x => x.Value.Age <= _maxAge).ToList();

            // Update existing tracked objects and add new untracked objects.
            if (activeTracks.Count > 0)
            {
                var costMatrix = CalculateCostMatrix(activeTracks, detections);

                // Match detected objects with tracked objects
                var assignedIds = MatchPredictedObjects(detections, activeTracks, costMatrix);

                // Add untracked new objects to tracker
                AddUntrackedObjects(detections, assignedIds);
            }
            else
            {
                // Tracker is empty; create new tracks for all current detections.
                CreateInitialTracks(detections);
            }
            
            RemoveOldTrackedObjects();
        }

        /// <summary>
        /// Adds new tracks for detections that were not matched to any existing tracked object.
        /// Called after the assignment phase.
        /// </summary>
        private void AddUntrackedObjects<T>(List<T> detections, HashSet<int> assignedDetections) where T : IDetection
        {
            // Add new untracked objects     
            for (int i = 0; i < detections.Count; i++)
            {
                if (assignedDetections.Contains(i) is false)
                {
                    _nextId++;

                    detections[i].Id = _nextId;
                    _trackedObjects[_nextId] = new TrackedObject(detections[i], _tailLength);
                }
            }
        }

        /// <summary>
        /// Adds new untracked objects to the tracker.  
        /// Used when the tracker is empty and all detections are considered new.
        /// </summary>
        public void CreateInitialTracks<T>(List<T> detections) where T : IDetection
        {
            foreach (var detection in detections)
            {
                _nextId++;

                detection.Id = _nextId;
                _trackedObjects[_nextId] = new TrackedObject(detection, _tailLength);
            }
        }

        /// <summary>
        /// Removes tracked objects that have not been matched for more than <c>_maxAge</c> frames.
        /// </summary>
        private void RemoveOldTrackedObjects()
        {
            foreach (var track in _trackedObjects.ToList())
            {
                track.Value.Age++;

                if (track.Value.Age > _maxAge)
                {
                    _trackedObjects.Remove(track.Key);
                }
            }
        }

        /// <summary>
        /// Computes the cost matrix between active tracks and current detections,
        /// based on IoU and normalized center-point distance.
        /// </summary>
        private static float[,] CalculateCostMatrix<T>(List<KeyValuePair<int, TrackedObject>> activeTracks, List<T> detections) where T : IDetection
        {
            // Define array for storing costMatrix
            var costMatrix = new float[activeTracks.Count, detections.Count];

            for (int i = 0; i < activeTracks.Count; i++)
            {
                var track = activeTracks[i].Value;
                var predictedState = track.GetPredictedState();

                float predictedX = predictedState[0];
                float predictedY = predictedState[1];

                for (int j = 0; j < detections.Count; j++)
                {
                    var detectionBox = detections[j].BoundingBox;
                    var iouCost = 1 - YoloCore.CalculateIoU(track.BoundingBox.BoundingBox, detectionBox);

                    var distance = CalculateDistance(predictedX, predictedY, detectionBox.MidX, detectionBox.MidY);
                    var distanceCost = distance / 100.0f; // Normalize distance

                    costMatrix[i, j] = iouCost + distanceCost;
                }
            }

            return costMatrix;
        }

        /// <summary>
        /// Matches current detections to existing active tracks using the LAPJV assignment algorithm
        /// based on the provided cost matrix. Updates tracked objects for matched detections,
        /// resets their age, and returns the set of assigned detection indices.
        /// </summary>
        private HashSet<int> MatchPredictedObjects<T>(List<T> detections, List<KeyValuePair<int, TrackedObject>> activeTracks, float[,] costMatrix) where T : IDetection
        {
            // Solve assignment using LAPJV algorithm.
            float[,] originalCostMatrix = (float[,])costMatrix.Clone();
            var assignment = LAPJV.Solve(costMatrix);

            // Assign detections to existing tracks
            var assignedDetections = new HashSet<int>();

            for (int i = 0; i < assignment.Length; i++)
            {
                var trackId = activeTracks[i].Key;
                var trackedBox = activeTracks[i].Value;

                int detectionIndex = assignment[i];

                // If a tracked Id is not found, add to age and remove if it's not detected in the last x frames
                if (detectionIndex == -1)
                    continue;

                var cost = originalCostMatrix[i, detectionIndex];

                if (cost < _costThreshold) // Instead of < 0.2f
                {
                    var box = detections[detectionIndex];

                    // Set Id of detected boundingbox
                    box.Id = trackId;

                    // Reset tracked box age counter
                    trackedBox.Age = 0;
                    trackedBox.TrackBoundingBox(box);

                    assignedDetections.Add(detectionIndex);
                }
            }

            return assignedDetections;
        }
        /// <summary>
        /// Calculate distance between two bounding box centers.
        /// </summary>
        private static float CalculateDistance(float predictedX, float predictedY, float detectionX, float detectionY)
        {
            var dx = predictedX - detectionX;
            var dy = predictedY - detectionY;
            return (float)Math.Sqrt(dx * dx + dy * dy); // Euclidean distance
        }
    }
}
