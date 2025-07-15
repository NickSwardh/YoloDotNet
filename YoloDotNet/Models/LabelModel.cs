namespace YoloDotNet.Models
{
    /// <summary>
    /// Represents a label with its associated color in hexadecimal format.
    /// </summary>
    public struct LabelModel
    {
        public LabelModel(int index, string name, string color)
        {
            this.Index = index;
            this.Name = name;
            this.Color = color;
        }

        /// <summary>
        /// Label index
        /// </summary>
        public int Index { get; init; }

        /// <summary>
        /// Name of the label.
        /// </summary>
        public string Name { get; init; } = default!;

        /// <summary>
        /// Label color in hexadecimal format.
        /// </summary>
        public string Color { get; init; } = default!;
    }
}
