namespace YoloDotNet.Trackers
{
    public partial class SortTracker
    {
        private readonly Dictionary<int, TrackedObject> _trackedObjects = [];

        private bool _trackerWasInitialized = false;
        private int _nextId = 0;
        private readonly int _maxAge;
        private readonly float _costThreshold;
        private readonly int _tailLength;

        public SortTracker(float costThreshold = 0.5f, int maxAge = 3, int tailLength = 30)
        {
            _costThreshold = costThreshold;
            _maxAge = maxAge + 1;
            _tailLength = tailLength;
        }

        public void UpdateTracker<T>(List<T> detections) where T : IDetection
        {
            // Initialize tracker on first tracked objects.
            if (_trackerWasInitialized is false)
            {
                _trackerWasInitialized = true;
                InitializeTracker(detections);
                return;
            }

            // If there is nothing to track, no further processing needed...
            if (detections.Count == 0)
                return;

            // Predict new positions using Kalman Filter
            foreach (var trackedObject in _trackedObjects.Values)
                trackedObject.KalmanPredict();

            // Get previous boundingboxes based on previous _trackCounter
            var activeTracks = _trackedObjects.Where(x => x.Value.Age < _maxAge).ToList();

            var costMatrix = CalculateCostMatrix(activeTracks, detections);

            // Match detected objects with tracked objects
            var assignedIds = MatchPredictedObjects(detections, activeTracks, costMatrix);

            // Add untracked new objects to tracker
            AddUntrackedObjects(detections, assignedIds);
            RemoveOldTrackedObjects();
        }

        private void AddUntrackedObjects<T>(List<T> detections, HashSet<int> assignedDetections) where T : IDetection
        {
            // Add new untracked objects     
            for (int i = 0; i < detections.Count; i++)
            {
                if (assignedDetections.Contains(i) is false)
                {
                    int newId = _nextId++;
                    detections[i].Id = newId;
                    _trackedObjects[newId] = new TrackedObject(detections[i], _tailLength);
                }
            }
        }

        private void RemoveOldTrackedObjects()
        {
            foreach (var track in _trackedObjects.ToList())
            {
                track.Value.Age++;

                if (track.Value.Age >= _maxAge)
                {
                    _trackedObjects.Remove(track.Key);
                }
            }
        }

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

                    trackedBox.Age = 0;
                    trackedBox.TrackBoundingBox(box);

                    assignedDetections.Add(detectionIndex);
                }
            }

            return assignedDetections;
        }

        public void InitializeTracker<T>(List<T> detections) where T : IDetection
        {
            foreach (var detection in detections)
            {
                _nextId++;

                detection.Id = _nextId;
                _trackedObjects[_nextId] = new TrackedObject(detection);
            }
        }

        // Calculate distance between two bounding box centers.
        private static float CalculateDistance(float predictedX, float predictedY, float detectionX, float detectionY)
        {
            var dx = predictedX - detectionX;
            var dy = predictedY - detectionY;
            return (float)Math.Sqrt(dx * dx + dy * dy); // Euclidean distance
        }
    }
}
