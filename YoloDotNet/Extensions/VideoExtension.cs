using System.Globalization;
using System.Text.RegularExpressions;
using YoloDotNet.Enums;
using YoloDotNet.Models;

namespace YoloDotNet.Extensions
{
    public static class VideoExtension
    {
        /// <summary>
        /// Get video info from extracted metadata
        /// </summary>
        /// <param name="grp"></param>
        /// <param name="_videoOptions"></param>
        /// <returns>VideoMetaData</returns>
        public static VideoMetaData BuildMetaData(this GroupCollection grp, VideoOptions _videoOptions)
        {
            var actualFps = double.Parse(grp["FPS"].Value, CultureInfo.InvariantCulture);
            var fps = _videoOptions.FPS ?? actualFps;
            var frames = int.Parse(grp["Frames"].Value);

            if (_videoOptions.FPS is not null)
                frames = (int)(frames / actualFps * _videoOptions.FPS);

            var width = _videoOptions.Width ?? int.Parse(grp["Width"].Value);
            var height = _videoOptions.Height ?? int.Parse(grp["Height"].Value);

            // By setting to -2, ffmpeg will automatically calculate with or hight to be dividable by 2
            if (_videoOptions.Width is not null && _videoOptions.Height is null)
                height = -2;

            if (_videoOptions.Height is not null && _videoOptions.Width is null)
                width = -2;

            return new VideoMetaData()
            {
                VideoFile = _videoOptions.VideoFile,
                OutputFolder = CreateOutputFolder(_videoOptions.OutputDir),
                TempFolder = CreateOutputFolder(Path.Combine(_videoOptions.OutputDir, nameof(FolderName.Temp)), true),
                Duration = TimeSpan.ParseExact(grp["Duration"].Value, "hh\\:mm\\:ss\\.ff", CultureInfo.InvariantCulture),
                FPS = fps,
                Frames = frames,
                Width = width,
                Height = height
            };
        }

        /// <summary>
        /// Manage temporary folder
        /// </summary>
        /// <param name="outputDir"></param>
        /// <param name="deleteOld">Delete everything? Delete folder if it exists and create a new folder - false by default.</param>
        /// <returns></returns>
        private static DirectoryInfo CreateOutputFolder(string outputDir, bool deleteOld = false)
        {
            if (deleteOld is true && Directory.Exists(outputDir))
                Directory.Delete(outputDir, true);

            return Directory.CreateDirectory(outputDir);
        }
    }
}
