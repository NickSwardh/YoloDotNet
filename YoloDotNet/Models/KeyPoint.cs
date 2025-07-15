namespace YoloDotNet.Models
{
    public struct KeyPoint
    {
        public KeyPoint(int x, int y, double confidence)
        {
            this.X = x;
            this.Y = y;
            this.Confidence = confidence;
        }

        public int X { get; }
        public int Y { get; }
        public double Confidence { get; }
    }

}
