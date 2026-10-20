// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Models
{
    public class PoseDrawingOptions : CommonBoundingBoxOptions
    {
        /// <summary>
        /// Confidence threshold for displaying keypoints.
        /// </summary>
        public double PoseConfidence { get; set; } = ImageConfig.POSE_KEYPOINT_THRESHOLD;

        /// <summary>
        /// Color, including its alpha, used to draw keypoints when no <see cref="PoseKeyPoints"/> are set, keypoints and
        /// connections the <see cref="PoseKeyPoints"/> define without a color, and keypoints beyond the last one defined.
        /// </summary>
        public SKColor DefaultPoseColor { get; set; } = ImageConfig.PoseMarkerColor;

        /// <summary>
        /// Defines how keypoints are colored and connected, loaded once at startup with
        /// <see cref="Models.PoseKeyPoints.Load(string)"/>. Without pose keypoints, keypoints are drawn as plain dots
        /// in <see cref="DefaultPoseColor"/> when bounding boxes, labels and tails are all disabled.
        /// <para>A keypoint count that differs from the model's is tolerated: extra keypoints are drawn with
        /// <see cref="DefaultPoseColor"/> and connections to keypoints the result lacks are skipped. Call
        /// <see cref="Models.PoseKeyPoints.ValidateKeyPointCount"/> at startup to fail early instead.</para>
        /// </summary>
        public PoseKeyPoints? PoseKeyPoints { get; set; }

        /// <summary>
        /// <b>Obsolete, has no effect.</b> Keypoint markers are ignored when drawing. Use <see cref="PoseKeyPoints"/> instead,
        /// e.g. <c>PoseKeyPoints = PoseKeyPoints.Load("pose-keypoints.json")</c>.
        /// </summary>
        [Obsolete("KeyPointMarkers has no effect and will be removed. Use PoseKeyPoints instead, e.g. PoseKeyPoints = PoseKeyPoints.Load(\"pose-keypoints.json\").")]
        public KeyPointMarker[] KeyPointMarkers { get; set; } = [];

        #region Mapping method
        public static explicit operator DetectionDrawingOptions(PoseDrawingOptions options) => new()
        {
            Font = options.Font,
            FontColor = options.FontColor,
            FontSize = options.FontSize,
            EnableFontShadow = options.EnableFontShadow,
            EnableDynamicScaling = options.EnableDynamicScaling,
            DrawLabelBackground = options.DrawLabelBackground,
            DrawBoundingBoxes = options.DrawBoundingBoxes,
            DrawLabels = options.DrawLabels,
            DrawConfidenceScore = options.DrawConfidenceScore,
            DrawTrackedTail = options.DrawTrackedTail,
            BorderThickness = options.BorderThickness,
            BoundingBoxOpacity = options.BoundingBoxOpacity,
            BoundingBoxHexColors = options.BoundingBoxHexColors,
            TailThickness = options.TailThickness,
            TailPaintColorStart = options.TailPaintColorStart,
            TailPaintColorEnd = options.TailPaintColorEnd,
            LabelPosition = options.LabelPosition,
        };
        #endregion
    }

    /// <summary>
    /// <b>Obsolete, has no effect.</b> Use <see cref="PoseKeyPoints"/> with PoseDrawingOptions.PoseKeyPoints instead.
    /// </summary>
    [Obsolete("KeyPointMarker has no effect and will be removed. Use PoseKeyPoints with PoseDrawingOptions.PoseKeyPoints instead.")]
    public record KeyPointMarker
    {
        /// <summary>
        /// Color associated with the keypoint.
        /// </summary>
        public string Color { get; init; } = default!;

        /// <summary>
        /// Defines connections between keypoints.
        /// </summary>
        public KeyPointConnection[] Connections { get; init; } = [];
    }

    /// <summary>
    /// <b>Obsolete, has no effect.</b> Use <see cref="PoseKeyPoints"/> with PoseDrawingOptions.PoseKeyPoints instead.
    /// </summary>
    [Obsolete("KeyPointConnection has no effect and will be removed. Use PoseKeyPoints with PoseDrawingOptions.PoseKeyPoints instead.")]
    public record KeyPointConnection(int Index, string Color);
}
