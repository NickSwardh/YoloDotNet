namespace YoloDotNet.Services
{
    internal class FFMPEGService : IDisposable
    {
        /// <summary>
        /// Mandatory callback that is invoked synchronously
        /// for every decoded frame. The loop blocks until the
        /// callback returns.
        /// </summary>
        public Action<SKBitmap, long>? OnFrameReady { get; set; }

        private const string FFMPEG = "ffmpeg";
        private const string FFPROBE = "ffprobe";

        private Process _ffmpegDecode = default!;
        private Process _ffmpegEncode = default!;

        public VideoOptions _options = default!;
        private CancellationTokenSource _cts = default!;

        public VideoMetadata VideoMetadata { get; set; } = default!;

        private int _videoHeight;
        private int _videoWidth;
        private double _fps;

        public FFMPEGService(VideoOptions options)
        {
            _options = options;

            GetVideoSourceDimensions();
            InitializeFFMPEGDecode();
            InitializeFFMPEGEncode();
        }

        private void GetVideoSourceDimensions()
        {
            var metadata = GetVideoInfo(_options.VideoFile);

            var (newWidth, newHeight) = CalculateProportionalResize(metadata, _options);

            _videoWidth = newWidth;
            _videoHeight = newHeight;
            _fps = (_options.FPS > 0) ? _options.FPS : metadata.FPS;

            // Give user metadata info about selected video
            VideoMetadata = new VideoMetadata(
                metadata.Width,
                metadata.Height,
                newWidth,
                newHeight,
                metadata.Duration,
                metadata.FPS,
                _options.FPS,
                metadata.TotalFrames,
                CalculateTargetFramesCount(metadata));
        }

        /// <summary>
        /// Pre-process video file by creating a temporary file with video stream only, in order to get actual duration of video and other video info
        /// </summary>
        public static Metadata GetVideoInfo(string videoPath)
        {
            using var ffprobe = Processor.Create(FFPROBE, [
                "-v",                       "quiet",
                "-print_format",            "json",
                "-select_streams",          "v:0",
                "-show_entries",            "stream=width,height,r_frame_rate,duration",
                "-hide_banner",
                $@"""{videoPath}"""]);

            ffprobe.Start();

            // Read standard output and error synchronously
            string output = ffprobe.StandardOutput.ReadToEnd();
            string error = ffprobe.StandardError.ReadToEnd();

            ffprobe.WaitForExit();

            using var doc = JsonDocument.Parse(output);

            if (doc.RootElement.TryGetProperty("streams", out var streams))
            {
                // Prepare json string
                var json = Regex.Replace(streams.ToString(),
                    @"""r_frame_rate"":\s""(\d+)\/(\d+)"",\s*""duration"":\s*""(\d+(?:\.\d+)?)""",
                    @"""frameratenumerator"": $1, ""frameratedenominator"": $2, ""duration"": $3");

                json = json.Replace("[", "")
                    .Replace("]", "");

                return JsonSerializer.Deserialize<Metadata>(json) ?? new();
            }

            return new();
        }

        private void InitializeFFMPEGDecode()
            => _ffmpegDecode = Processor.Create(FFMPEG, [
                "-i",                       $@"""{_options.VideoFile}""",
                "-an",
                "-vf",                      $@"""fps={_fps.ToString("", CultureInfo.InvariantCulture)},scale={_videoWidth}:{_videoHeight}""",
                //"-vf",                      $@"""select='not(mod(n,1000))',setpts=N/FRAME_RATE/TB,scale={_videoWidth}:{_videoHeight}""",
                "-pix_fmt",                 "bgra",
                "-vcodec",                  "rawvideo",
                "-f",                       "image2pipe",
                "-"]); // Pipe output to YoloDotNet.

        private void InitializeFFMPEGEncode()
            => _ffmpegEncode = Processor.Create(FFMPEG, [
                "-f",       "rawvideo",
                "-pix_fmt", "bgra",
                "-s",       $"{_videoWidth}:{_videoHeight}",
                "-framerate", $"{_fps.ToString("", CultureInfo.InvariantCulture)}",
                "-i",        "-", //  Pipe input from YoloDotNet.
                "-c:v",     "h264_nvenc",
                "-vf",      $@"""setsar=1:1""",
                "-rc:v:0",  "vbr_hq",
                "-cq:v",    $"{_options.Quality}",
                "-y",       $@"""{Path.Combine(_options.OutputDir, "output.mp4")}"""]);

        public void Start()
        {
            _cts = new CancellationTokenSource();
            Run(_cts.Token);
        }

        public void Stop()
            => _cts.Cancel();

        unsafe private void Run(CancellationToken cancellationToken)
        {
            var frameSize = _videoWidth * _videoHeight * 4;
            var buffer = new byte[frameSize];
            int frameIndex = 0;

            _ffmpegDecode.Start();
            _ffmpegDecode.BeginErrorReadLine();

            if (_options.GenerateVideo is true)
            {
                _ffmpegEncode.Start();
                _ffmpegEncode.BeginErrorReadLine();
            }

            using var inputStream = _ffmpegDecode.StandardOutput.BaseStream;
            using Stream? outputStream = (_options.GenerateVideo is true)
                ? _ffmpegEncode.StandardInput.BaseStream
                : default!;

            var frame = new SKBitmap(_videoWidth, _videoHeight, SKColorType.Bgra8888, SKAlphaType.Opaque);

            try
            {
                while (cancellationToken.IsCancellationRequested is false)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    int bytesRead = 0;

                    while (bytesRead < frameSize)
                    {
                        int read = inputStream.Read(buffer, bytesRead, frameSize - bytesRead);

                        // Exit if stream reached its end.
                        if (read == 0)
                            return;

                        bytesRead += read;
                    }

                    // Fill frame with pixels from ffmpeg
                    fixed (byte* ptr = buffer)
                    {
                        frame.SetPixels((IntPtr)ptr);
                    }

                    // Let user process the frame...
                    OnFrameReady?.Invoke(frame, frameIndex);

                    // Encode frame back to video?
                    if (_options.GenerateVideo is true)
                    {
                        outputStream.Write(frame.Bytes);
                    }

                    frameIndex++;
                }
            }
            catch (OperationCanceledException)
            {
                // Service stopped by user. Exit gracefully...
            }
            finally
            {
                inputStream?.Flush();
                inputStream?.Close();

                outputStream?.Flush();
                outputStream?.Close();

                _ffmpegDecode.WaitForExit();

                if (_options.GenerateVideo is true)
                    _ffmpegEncode.WaitForExit();
            }
        }

        private static (int width, int height) CalculateProportionalResize(Metadata metadata, VideoOptions options)
        {
            int originalWidth = metadata.Width;
            int originalHeight = metadata.Height;
            int targetWidth = options.Width;
            int targetHeight = options.Height;

            targetWidth = (targetWidth == -2 || options.Width > 0)
                ? options.Width
                : originalWidth;

            targetHeight = (targetHeight == -2 || options.Height > 0)
                ? options.Height
                : originalHeight;

            if (targetWidth == -2 && targetHeight == -2)
                throw new ArgumentException("Both with and height cant be -2.");

            if (targetWidth == -2)
            {
                // Calculate width proportionally based on target height
                float scale = (float)targetHeight / originalHeight;
                int newWidth = (int)(originalWidth * scale);
                if (newWidth % 2 != 0) newWidth--; // FFmpeg requires even
                return (newWidth, targetHeight);
            }

            if (targetHeight == -2)
            {
                // Calculate height proportionally based on target width
                float scale = (float)targetWidth / originalWidth;
                int newHeight = (int)(originalHeight * scale);
                if (newHeight % 2 != 0) newHeight--; // FFmpeg requires even
                return (targetWidth, newHeight);
            }

            // Both values are set → resize proportionally to fit in bounds
            float ratioX = (float)targetWidth / originalWidth;
            float ratioY = (float)targetHeight / originalHeight;
            float scaleRatio = Math.Min(ratioX, ratioY);

            int finalWidth = (int)(originalWidth * scaleRatio);
            int finalHeight = (int)(originalHeight * scaleRatio);

            // Ensure even dimensions
            if (finalWidth % 2 != 0) finalWidth--;
            if (finalHeight % 2 != 0) finalHeight--;

            return (finalWidth, finalHeight);
        }

        private long CalculateTargetFramesCount(Metadata metadata)
        {
            var targetFps = _fps;

            // Look for the decimal point
            string str = targetFps.ToString("G17", CultureInfo.InvariantCulture);
            var index = str.IndexOf('.');

            int decimalDigits = index > 0
                ? str.Length - index - 1
                : 0;

            long totalFrames;

            // If target fps is the same as original fps
            if (targetFps == Math.Round(metadata.FPS, decimalDigits))
            {
                totalFrames = (int)Math.Floor(targetFps * metadata.Duration) - 1;
            }
            else
            {
                var fps = metadata.FrameRateDenominator > 1000
                    ? targetFps * 1000 / metadata.FrameRateDenominator
                    : targetFps;

                totalFrames = (int)Math.Floor(fps * metadata.Duration) - 1;
            }

            return totalFrames - 1;
        }

        public void Dispose()
        {
            _cts?.Cancel();
            _cts?.Dispose();

            _ffmpegDecode?.Dispose();
            _ffmpegDecode?.Dispose();

            GC.SuppressFinalize(this);
        }
    }
}
