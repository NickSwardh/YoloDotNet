namespace YoloDotNet.Models.Interfaces
{
    public interface IDetection
    {
        LabelModel Label { get; init; }
        double Confidence { get; init; }
        SKRectI BoundingBox { get; init; }

        // Optional properties used for SORT Tracking
        int? Id { get; set; }

        List<SKPoint>? Tail { get; set; }
    }
}