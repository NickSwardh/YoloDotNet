namespace YoloDotNet.Models
{
    /// <summary>
    /// Represents a label with its associated color in hexadecimal format.
    /// </summary>
    public record LabelModel
    {
        /// <summary>
        /// Label index
        /// </summary>
        public int Index { get; init; }

        /// <summary>
        /// Name of the label.
        /// </summary>
        public string Name { get; init; } = default!;
    }
}
