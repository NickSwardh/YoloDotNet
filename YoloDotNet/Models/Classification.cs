using YoloDotNet.Attributes;

namespace YoloDotNet.Models
{
    [OutputShape]
    public class Classification
    {
        public string Label { get; set; } = default!;
        public double Confidence { get; set; }
    }
}
