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
        public static void ParseVideoMetaData(this VideoMetaData metaData, VideoSettings videoSettings)
        {
            var te = metaData.Streams[0].Framerate.Split('/');
            var framesPerSecond = int.Parse(te[0]);
            var rate = int.Parse(te[1]);
            var timeStamp = metaData.Streams[0].Duration;
            var width = metaData.Streams[0].Width;
            var height = metaData.Streams[0].Width;
            var duration = TimeSpan.FromSeconds(timeStamp);

            // Calculate actual fps
            double fps = framesPerSecond / rate;

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
