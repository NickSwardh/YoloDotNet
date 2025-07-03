﻿namespace YoloDotNet.Models
{
    /// <summary>
    /// Contains tracking-related metadata such as assigned ID and motion trail,
    /// available only when object tracking is enabled.
    /// </summary>
    [EditorBrowsable(EditorBrowsableState.Never)]
    public class TrackingInfo
    {
        /// <summary>
        /// The unique identifier assigned to the object by the tracker, if tracking is enabled.
        /// </summary>
        public int? Id { get; set; }

        /// <summary>
        /// A list of points representing the tail or path history of the tracked object, 
        /// useful for visualizing motion across frames.
        /// </summary>
        public List<SKPoint>? Tail { get; set; }
    }
}
