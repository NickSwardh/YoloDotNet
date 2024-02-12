namespace YoloDotNet.Models
{
    public interface IDetection
    {
        LabelModel Label { get; init; }
        double Confidence { get; init; }
        Rectangle Rectangle { get; init; }
    }
}