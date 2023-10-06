namespace YoloDotNet.Models
{
    /// <summary>
    /// Represents a label with its associated color in hexadecimal format.
    /// </summary>
    /// <remarks>
    public record LabelModel
    {
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
