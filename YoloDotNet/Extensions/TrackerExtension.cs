namespace YoloDotNet.Extensions
{
    public static class TrackerExtension
    {
        public static List<T> Track<T>(this List<T> detections, SortTrack sortTrack) where T : IDetection
        {
            // TODO: Rename method
            sortTrack.UpdateTracker(detections);
            return detections;
        }
    }
}
