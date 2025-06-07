namespace YoloDotNet.Trackers
{
    /// <summary>
    /// Primary constructor
    /// </summary>
    /// <param name="maxLength"></param>
    public class TailTrack(int maxLength)
    {
        private readonly int _maxLength = maxLength;
        private readonly Queue<SKPoint> _positions = new (maxLength);

        public List<SKPoint> GetTail() => [.. _positions];

        public void AddTailPoint(SKPoint point)
        {
            RemoveOldestTailPoint();

            _positions.Enqueue(point);
        }

        private void RemoveOldestTailPoint()
        {
            var len = _positions.Count;

            if (_positions.Count >= _maxLength)
                _positions.Dequeue(); // Remove oldest
        }
    }
}
