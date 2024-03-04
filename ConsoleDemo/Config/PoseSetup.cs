using YoloDotNet.Models;

namespace ConsoleDemo.Config
{
    /// <summary>
    /// Demonstrates configuring a custom pose marker profile with personalized colors and illustrating marker connections.
    /// </summary>
    public static class CustomPoseMarkerColorMap
    {
        /// <summary>
        /// Pose markers in the same EXACT order(!) as in the trained model
        /// </summary>
        private enum PoseMarker
        {
            Nose,
            LeftEye,
            RightEye,
            LeftEar,
            RightEar,
            LeftShoulder,
            RightShoulder,
            LeftElbow,
            RightElbow,
            LeftWrist,
            RightWrist,
            LeftHip,
            RightHip,
            LeftKnee,
            RightKnee,
            LeftAnkle,
            RightAnkle
        }

        /// <summary>
        /// Color names for identifying hexadecimal colors
        /// </summary>
        private enum PoseColor
        {
            Green,
            LightBlue,
            Yellow,
            HotPink
        }

        /// <summary>
        /// Named hexadecimal colors
        /// </summary>
        private static Dictionary<PoseColor, string> Colors => new()
        {
            { PoseColor.Green, "#A2FF33" },     // Light green
            { PoseColor.LightBlue, "#33ACFF" }, // Light blue
            { PoseColor.Yellow, "#FFF633" },    // Yellow
            { PoseColor.HotPink, "#FF33AC" }    // Hot pink
        };

        /// <summary>
        /// Options for pose estimation.
        /// </summary>
        public static PoseOptions PoseMarkerOptions => new()
        {
            PoseConfidence = 0.65,
            PoseMarkers = GetPoseMarkerConfiguration,
            DrawBoundingBox = true
        };

        #region Method for configuring custom pose-markers and their connections
        /// <summary>
        /// Configure the connections between pose markers and specify the colors to use.
        /// </summary>
        private static YoloDotNet.Models.PoseMarker[] GetPoseMarkerConfiguration =>
        [
            new () // Nose
            {
                Color = Colors[PoseColor.Green],
                Connections =
                [
                    new ((int)PoseMarker.LeftEye, Colors[PoseColor.Green]),
                    new ((int)PoseMarker.RightEye, Colors[PoseColor.Green])
                ]
            },
            new () // Left eye
            {
                Color = Colors[PoseColor.Green],
                Connections = [ new ((int)PoseMarker.RightEye, Colors[PoseColor.Green]) ]
            },
            new () // Right eye
            {
                Color = Colors[PoseColor.Green],
            },
            new () // Left ear
            {
                Color = Colors[PoseColor.Green],
                Connections =
                [
                    new ((int)PoseMarker.LeftEye, Colors[PoseColor.Green]),
                    new ((int)PoseMarker.LeftShoulder, Colors[PoseColor.Green]),
                ]
            },
            new () // Right ear
            {
                Color = Colors[PoseColor.Green],
                Connections =
                [
                    new ((int)PoseMarker.RightEye, Colors[PoseColor.Green]),
                    new ((int)PoseMarker.RightShoulder, Colors[PoseColor.Green]),
                ]
            },
            new () // Left shoulder
            {
                Color = Colors[PoseColor.LightBlue],
                Connections =
                [
                    new ((int)PoseMarker.RightShoulder, Colors[PoseColor.LightBlue]),
                    new ((int)PoseMarker.LeftElbow, Colors[PoseColor.LightBlue]),
                    new ((int)PoseMarker.LeftHip, Colors[PoseColor.HotPink])
                ]
            },
            new () // Right shoulder
            {
                Color = Colors[PoseColor.LightBlue],
                Connections =
                [
                    new ((int)PoseMarker.RightElbow, Colors[PoseColor.LightBlue]),
                    new ((int)PoseMarker.RightHip, Colors[PoseColor.HotPink])
                ]
            },
            new () // Left elbow
            {
                Color = Colors[PoseColor.LightBlue],
                Connections = [ new ((int)PoseMarker.LeftWrist, Colors[PoseColor.LightBlue]) ]
            },
            new () // Right elbow
            {
                Color = Colors[PoseColor.LightBlue],
                Connections = [ new ((int)PoseMarker.RightWrist, Colors[PoseColor.LightBlue]) ]
            },
            new () // Left wrist
            {
                Color = Colors[PoseColor.LightBlue]
            },
            new () // Right wrist
            {
                Color = Colors[PoseColor.LightBlue]
            },
            new () // Left hip
            {
                Color = Colors[PoseColor.Yellow],
                Connections =
                [
                    new ((int)PoseMarker.RightHip, Colors[PoseColor.HotPink]),
                    new ((int)PoseMarker.LeftKnee, Colors[PoseColor.Yellow])
                ]
            },
            new () // Right hip
            {
                Color = Colors[PoseColor.Yellow],
                Connections = [ new ((int)PoseMarker.RightKnee, Colors[PoseColor.Yellow]) ]
            },
            new () // Left knee
            {
                Color = Colors[PoseColor.Yellow],
                Connections = [ new ((int)PoseMarker.LeftAnkle, Colors[PoseColor.Yellow]) ]
            },
            new () // Right knee
            {
                Color = Colors[PoseColor.Yellow],
                Connections = [ new ((int)PoseMarker.RightAnkle, Colors[PoseColor.Yellow]) ]
            },
            new () // Left ankle
            {
                Color = Colors[PoseColor.Yellow]
            },
            new () // Right ankle
            {
                Color = Colors[PoseColor.Yellow]
            }
        ];
        #endregion

    }
}