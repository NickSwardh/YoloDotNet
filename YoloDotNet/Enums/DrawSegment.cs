namespace YoloDotNet.Enums
{
    public enum DrawSegment
    {
        /// <summary>
        /// Draw both boundingbox and pixelmask
        /// </summary>
        Default,

        /// <summary>
        /// Only draw Boundingbox
        /// </summary>
        BoundingBoxOnly,

        /// <summary>
        /// Only draw pixel-mask
        /// </summary>
        PixelMaskOnly,
    }
}
