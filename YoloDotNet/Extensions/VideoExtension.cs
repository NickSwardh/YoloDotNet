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
        public static void ParseVideoMetaData(this GroupCollection grp, VideoSettings videoSettings)
        {
            // Parse metadata
            var frames = double.Parse(grp["Frame"].Value, CultureInfo.InvariantCulture);
            var rate = double.Parse(grp["Rate"].Value, CultureInfo.InvariantCulture);
            var timeStamp = double.Parse(grp["Duration"].Value, CultureInfo.InvariantCulture);
            var width = videoSettings.Width ?? int.Parse(grp["Width"].Value);
            var height = videoSettings.Height ?? int.Parse(grp["Height"].Value);
            var duration = TimeSpan.FromSeconds(timeStamp);

            // Calculate actual fps
            var fps = frames / rate;

            // Calculate total frames
            var totalFrames = (int)Math.Round(fps * timeStamp);

            // If a custom FPS has been set by user, set new FPS and re-calculate total frames
            if (videoSettings.FPS is not null)
            {
                fps = (double)videoSettings.FPS;
                totalFrames = (int)(totalFrames / fps * videoSettings.FPS);
            }

            // By setting resolution to -2, ffmpeg will automatically calculate width or height to be dividable by 2
            if (videoSettings.Width is not null && videoSettings.Height is null)
                height = -2;

            if (videoSettings.Height is not null && videoSettings.Width is null)
                width = -2;

            // Update video settings
            videoSettings.Duration = duration;
            videoSettings.FPS = fps;
            videoSettings.TotalFrames = totalFrames;
            videoSettings.Width = width;
            videoSettings.Height = height;
        }

        /// <summary>
        /// Manage temporary folder
        /// </summary>
        /// <param name="outputDir"></param>
        /// <param name="deleteOld">Delete everything? Delete folder if it exists and create a new folder - false by default.</param>
        /// <returns></returns>
        public static string CreateOutputFolder(string outputDir, bool deleteOld = false)
        {
            var dir = new DirectoryInfo(outputDir);

            if (dir.Exists && deleteOld is true)
                dir.Delete(true);

            var maxTries = 3;

            while (dir.Exists is false && maxTries > 0)
            {
                try
                {
                    dir.Create();
                }
                catch (Exception)
                {
                    if (maxTries == 1)
                        throw;

                    Thread.Sleep(1000);
                }
                
                maxTries--;
            }

            return dir.FullName;
        }

        /// <summary>
        /// Delete a file
        /// </summary>
        /// <param name="filePath"></param>
        public static void DeleteFile(this string filePath)
        {
            if (File.Exists(filePath) is false)
                return;

            File.Delete(filePath);
        }
    }
}
