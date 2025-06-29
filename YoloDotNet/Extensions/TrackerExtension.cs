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
