namespace YoloDotNet.Trackers
{
    public class TrackedObject(IDetection boundingBox, int tailLength = 30)
    {
        public IDetection BoundingBox { get; private set; } = boundingBox;
        public int Age { get; set; } = 0;
        public KalmanFilter Kalman { get; private set; } = new KalmanFilter(boundingBox.BoundingBox.MidX, boundingBox.BoundingBox.MidY);

        private readonly TailTrack _tailTracker = new(tailLength);

        public void TrackBoundingBox(IDetection detection)
        {
            // Update boundingbox
            BoundingBox = detection;

            // Store current boundingbox center coordinate
            var box = detection.BoundingBox;

            // Update tail
            detection.Tail = _tailTracker.GetTail();

            _tailTracker.AddTailPoint(new SKPointI(box.MidX, box.MidY)); // Store center of boundingbox.

            // Update Kalman filter
            Kalman.Update(box.MidX, box.MidY);
        }

        public void KalmanPredict()
            => Kalman.Predict();

        public float[] GetPredictedState()
            => Kalman.GetState();
    }
}
