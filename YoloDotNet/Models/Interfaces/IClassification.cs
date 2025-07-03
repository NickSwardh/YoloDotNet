namespace YoloDotNet.Models.Interfaces
{
    public interface IClassification
    {
        string Label { get; set; }
        double Confidence { get; set; }
    }
}
