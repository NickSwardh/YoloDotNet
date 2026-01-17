// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Benchmarks.Setup
{
    /// <summary>
    /// Demonstrates configuring a custom keypoint marker profile with custom colors and illustrating keypoint connections.
    /// </summary>
    public static class CustomKeyPointColorMap
    {
        /// <summary>
        /// Keypoints must be in the SAME EXACT order(!) as the classes in your trained model.
        /// </summary>
        private enum KeyPointType
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
        private enum KeyPointColor
        {
            Green,
            LightBlue,
            Yellow,
            HotPink
        }

        /// <summary>
        /// Named hexadecimal colors
        /// </summary>
        private static Dictionary<KeyPointColor, string> Colors => new()
        {
            { KeyPointColor.Green, "#A2FF33" },     // Light green
            { KeyPointColor.LightBlue, "#33ACFF" }, // Light blue
            { KeyPointColor.Yellow, "#FFF633" },    // Yellow
            { KeyPointColor.HotPink, "#FF33AC" }    // Hot pink
        };

        ///// <summary>
        ///// Keypoint options.
        ///// </summary>
        //public static PoseDrawingOptions KeyPointOptions => new()
        //{
        //    PoseConfidence = 0.65,
        //    KeyPointMarkers = KeyPointMapping
        //};

        #region Method for configuring custom keypoints and their connections
        /// <summary>
        /// Configure keypoint-connections and what colors to use.
        /// </summary>
        public static KeyPointMarker[] KeyPoints =>
        [
            new () // Nose
            {
                Color = Colors[KeyPointColor.Green],
                Connections =
                [
                    new ((int)KeyPointType.LeftEye, Colors[KeyPointColor.Green]),
                    new ((int)KeyPointType.RightEye, Colors[KeyPointColor.Green])
                ]
            },
            new () // Left eye
            {
                Color = Colors[KeyPointColor.Green],
                Connections = [ new ((int)KeyPointType.RightEye, Colors[KeyPointColor.Green]) ]
            },
            new () // Right eye
            {
                Color = Colors[KeyPointColor.Green],
            },
            new () // Left ear
            {
                Color = Colors[KeyPointColor.Green],
                Connections =
                [
                    new ((int)KeyPointType.LeftEye, Colors[KeyPointColor.Green]),
                    new ((int)KeyPointType.LeftShoulder, Colors[KeyPointColor.Green]),
                ]
            },
            new () // Right ear
            {
                Color = Colors[KeyPointColor.Green],
                Connections =
                [
                    new ((int)KeyPointType.RightEye, Colors[KeyPointColor.Green]),
                    new ((int)KeyPointType.RightShoulder, Colors[KeyPointColor.Green]),
                ]
            },
            new () // Left shoulder
            {
                Color = Colors[KeyPointColor.LightBlue],
                Connections =
                [
                    new ((int)KeyPointType.RightShoulder, Colors[KeyPointColor.LightBlue]),
                    new ((int)KeyPointType.LeftElbow, Colors[KeyPointColor.LightBlue]),
                    new ((int)KeyPointType.LeftHip, Colors[KeyPointColor.HotPink])
                ]
            },
            new () // Right shoulder
            {
                Color = Colors[KeyPointColor.LightBlue],
                Connections =
                [
                    new ((int)KeyPointType.RightElbow, Colors[KeyPointColor.LightBlue]),
                    new ((int)KeyPointType.RightHip, Colors[KeyPointColor.HotPink])
                ]
            },
            new () // Left elbow
            {
                Color = Colors[KeyPointColor.LightBlue],
                Connections = [ new ((int)KeyPointType.LeftWrist, Colors[KeyPointColor.LightBlue]) ]
            },
            new () // Right elbow
            {
                Color = Colors[KeyPointColor.LightBlue],
                Connections = [ new ((int)KeyPointType.RightWrist, Colors[KeyPointColor.LightBlue]) ]
            },
            new () // Left wrist
            {
                Color = Colors[KeyPointColor.LightBlue]
            },
            new () // Right wrist
            {
                Color = Colors[KeyPointColor.LightBlue]
            },
            new () // Left hip
            {
                Color = Colors[KeyPointColor.Yellow],
                Connections =
                [
                    new ((int)KeyPointType.RightHip, Colors[KeyPointColor.HotPink]),
                    new ((int)KeyPointType.LeftKnee, Colors[KeyPointColor.Yellow])
                ]
            },
            new () // Right hip
            {
                Color = Colors[KeyPointColor.Yellow],
                Connections = [ new ((int)KeyPointType.RightKnee, Colors[KeyPointColor.Yellow]) ]
            },
            new () // Left knee
            {
                Color = Colors[KeyPointColor.Yellow],
                Connections = [ new ((int)KeyPointType.LeftAnkle, Colors[KeyPointColor.Yellow]) ]
            },
            new () // Right knee
            {
                Color = Colors[KeyPointColor.Yellow],
                Connections = [ new ((int)KeyPointType.RightAnkle, Colors[KeyPointColor.Yellow]) ]
            },
            new () // Left ankle
            {
                Color = Colors[KeyPointColor.Yellow]
            },
            new () // Right ankle
            {
                Color = Colors[KeyPointColor.Yellow]
            }
        ];
        #endregion

    }
}