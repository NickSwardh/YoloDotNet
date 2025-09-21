// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Video.Services
{
    internal class FFmpegService : IDisposable
    {
        /// <summary>
        /// Mandatory callback that is invoked synchronously
        /// for every decoded frame. The loop blocks until the
        /// callback returns.
        /// </summary>
        public Action<SKBitmap, long>? OnFrameReady { get; set; }
        public Action? OnVideoEnd { get; set; }

        private const string FFMPEG = "ffmpeg";
        private const string FFPROBE = "ffprobe";

        private Process _ffmpegDecode = default!;
        private Process _ffmpegEncode = default!;

        public readonly VideoOptions _videoOptions = default!;
        private readonly YoloOptions _yoloOptions;
        private CancellationTokenSource _cts = default!;

        public VideoMetadata VideoMetadata { get; set; } = default!;

        private int _videoTargetHeight;
        private int _videoTargetWidth;
        private double _videoTargetfps;

        public FFmpegService(VideoOptions options, YoloOptions yoloOptions)
        {
            EnsureToolIsInstalled(FFPROBE);
            EnsureToolIsInstalled(FFMPEG);

            _videoOptions = options;
            _yoloOptions = yoloOptions;
            GetVideoSourceDimensions();
            InitializeFFMPEGDecode();
            InitializeFFMPEGEncode();
        }

        private void GetVideoSourceDimensions()
        {
            if (_videoOptions.VideoInput.StartsWith("device=", StringComparison.OrdinalIgnoreCase))
            {
                var (deviceName, width, height, fps) = GetDeviceInfo();

                _videoTargetfps = fps;
                (_videoTargetWidth, _videoTargetHeight) = CalculateProportionalResize(new Metadata { Width = width, Height = height }, _videoOptions);

                // Give user metadata info about selected video
                VideoMetadata = new VideoMetadata(
                    width,
                    height,
                    _videoTargetWidth,
                    _videoTargetHeight,
                    0,
                    _videoTargetfps,
                    _videoTargetfps,
                    0,
                    0,
                    deviceName);

                return;
            }

            var metadata = GetVideoInfo(_videoOptions.VideoInput);

            var (newWidth, newHeight) = CalculateProportionalResize(metadata, _videoOptions);

            _videoTargetWidth = newWidth;
            _videoTargetHeight = newHeight;
            _videoTargetfps = _videoOptions.FrameRate.Value != 0 ? _videoOptions.FrameRate.Value : metadata.FPS;

            // Give user metadata info about selected video
            VideoMetadata = new VideoMetadata(
                metadata.Width,
                metadata.Height,
                newWidth,
                newHeight,
                metadata.Duration,
                metadata.FPS,
                _videoOptions.FrameRate,
                metadata.TotalFrames,
                CalculateTargetFramesCount(metadata));
        }

        public (string, int, int, float) GetDeviceInfo()
        {
            try
            {
                var device = _videoOptions.VideoInput.Replace("device=", "");
                var deviceInfo = device.Split(':');

                var deviceName = deviceInfo[0].Trim();
                var width = int.Parse(deviceInfo[1]);
                var height = int.Parse(deviceInfo[2]);
                var fps = float.Parse(deviceInfo[3]);

                return (deviceName, width, height, fps);
            }
            catch (Exception)
            {
                throw new YoloDotNetVideoException(
                    $"Invalid video device input format: '{_videoOptions.VideoInput}'.\n" +
                    $"Expected format: 'DeviceName:Width:Height:FPS'.\n" +
                    $"Each part must be separated by a colon and must include:\n" +
                    $"  - DeviceName (e.g., Logitech BRIO)\n" +
                    $"  - Width (e.g., 1280)\n" +
                    $"  - Height (e.g., 720)\n" +
                    $"  - FPS (e.g., 30)\n" +
                    $"Example: 'Logitech BRIO:1280:720:30'.",
                    nameof(_videoOptions.VideoInput));
            }
        }

        /// <summary>
        /// Pre-process video file by creating a temporary file with video stream only, in order to get actual duration of video and other video info
        /// </summary>
        public static Metadata GetVideoInfo(string videoPath)
        {
            using var ffprobe = Processor.Create(FFPROBE, [
                "-v",                   "quiet",
                "-print_format",        "json",
                "-select_streams",      "v:0",
                "-show_entries",        "stream=width,height,r_frame_rate,duration",
                "-hide_banner",         videoPath]);

            ffprobe.Start();

            // Read standard output and error synchronously
            string output = ffprobe.StandardOutput.ReadToEnd();
            string error = ffprobe.StandardError.ReadToEnd();

            ffprobe.WaitForExit();

            using var doc = JsonDocument.Parse(output);

            if (doc.RootElement.TryGetProperty("streams", out var streams))
            {
                var test = streams.ToString();

                // Prepare json string
                var json = Regex.Replace(streams.ToString(),
                    @"""r_frame_rate"":\s""(\d+)\/(\d+)""",
                    @"""frameratenumerator"": $1,""frameratedenominator"": $2");

                json = Regex.Replace(json,
                    @"(,\s*""duration"":\s*)""([1-9](?:\d+)?(?:\.\d+)?)""", "$1$2", RegexOptions.Singleline);

                json = json.Replace("[", "").Replace("]", "");

                return JsonSerializer.Deserialize<Metadata>(json) ?? new();
            }
            else
                throw new YoloDotNetVideoException("The specified video stream is invalid and could not be processed. ", nameof(_videoOptions.VideoInput));
        }

        private void InitializeFFMPEGDecode()
        {
            var ffmpegArgs = new List<string>();

            string inputSource = _videoOptions.VideoInput;

            if (string.IsNullOrEmpty(VideoMetadata.DeviceName) is false)
            {
                // Select the correct input format based on platform
                string? deviceVideoFilter;
                if (SystemPlatform.GetOS() == Platform.Windows)
                {
                    deviceVideoFilter = "dshow"; // windows: Directshow
                    inputSource = $"video={VideoMetadata.DeviceName}";
                }
                else
                {
                    deviceVideoFilter = "v4l2"; // linux: Video4Linux2
                    inputSource = $"/dev/{VideoMetadata.DeviceName}"; // e.g., "video0"
                }

                ffmpegArgs.AddRange([
                    "-f",               deviceVideoFilter,
                    "-video_size",     $"{VideoMetadata.Width}x{VideoMetadata.Height}"]); // Force device to use full resolution
            }

            // Process all frames or every nth frame?
            var videoFilter = _videoOptions.FrameInterval <= 0
                ? $@"fps={_videoTargetfps.ToString(CultureInfo.InvariantCulture)},scale={_videoTargetWidth}:{_videoTargetHeight}"
                : $@"select='not(mod(n,{_videoOptions.FrameInterval}))',setpts=N/FRAME_RATE/TB,scale={_videoTargetWidth}:{_videoTargetHeight}";

            ffmpegArgs.AddRange([
                "-i",           inputSource,
                "-an",
                "-vf",           videoFilter,
                "-pix_fmt",     "bgra",
                "-vcodec",      "rawvideo",
                "-f",           "image2pipe",
                "-"]);          // Pipe output to YoloDotNet.

            _ffmpegDecode = Processor.Create(FFMPEG, ffmpegArgs);
        }

        private void InitializeFFMPEGEncode()
        {
            if (string.IsNullOrEmpty(_videoOptions.VideoOutput))
                return;

            // Pipe outgoing video from YoloDotNet
            var ffmpegArgs = new List<string>
            {
                "-f",       "rawvideo",
                "-pix_fmt", "bgra",
                "-s",       $"{_videoTargetWidth}:{_videoTargetHeight}",
            };

            var framerate = $"{_videoTargetfps.ToString("", CultureInfo.InvariantCulture)}";
            var vf = $@"setsar=1:1";

            if (_videoOptions.FrameInterval > 0)
            {
                framerate = $@"{(VideoMetadata.FPS / _videoOptions.FrameInterval).ToString("0.######", CultureInfo.InvariantCulture)}";
                vf = $@"fps={_videoTargetfps},setsar=1:1";
            }

            ffmpegArgs.AddRange([
                "-framerate",   framerate,
                "-i",           "-"]);

            // Video codec based on execution provider
            var videoCodec = _yoloOptions.ExecutionProvider switch
            {
                ICuda => "h264_nvenc",
                IOpenVino => "h264_qsv",
                _ => "libx264" // Fallback to CPU encoding
            };

            ffmpegArgs.AddRange([
                "-c:v",     videoCodec,
                "-vf",      vf,
                "-rc:v:0",  "vbr_hq",
                "-cq:v",    $"{_videoOptions.CompressionQuality}"]);

            // Split video in chunks?
            if (_videoOptions.VideoChunkDuration > 0)
            {
                var fullPath = Path.GetDirectoryName(_videoOptions.VideoOutput);
                var fileName = Path.GetFileNameWithoutExtension(_videoOptions.VideoOutput);
                var extension = Path.GetExtension(_videoOptions.VideoOutput);
                var videoOutput = Path.Combine(fullPath!, $"{fileName}_%d_{DateTime.Now:yyyyMMdd_hhmmss}{extension}");

                ffmpegArgs.AddRange([
                    "-g",               (_videoTargetfps * 2).ToString(),
                    "-segment_time",    _videoOptions.VideoChunkDuration.ToString(),
                    "-f",               "segment",
                    "-y",               videoOutput]);
            }
            else
            {
                ffmpegArgs.AddRange(["-y", _videoOptions.VideoOutput]);
            }

            _ffmpegEncode = Processor.Create(FFMPEG, ffmpegArgs);
        }

        public void Start()
        {
            _cts = new CancellationTokenSource();
            Run(_cts.Token);
        }

        public void Stop()
            => _cts.Cancel();

        unsafe private void Run(CancellationToken cancellationToken)
        {
            var frameSize = _videoTargetWidth * _videoTargetHeight * 4;
            var buffer = new byte[frameSize];
            int frameIndex = 0;

            var shouldCreateVideo = string.IsNullOrEmpty(_videoOptions.VideoOutput) is false;

            _ffmpegDecode.Start();
            _ffmpegDecode.BeginErrorReadLine();

            if (shouldCreateVideo)
            {
                _ffmpegEncode.Start();
                _ffmpegEncode.BeginErrorReadLine();
            }

            using var inputStream = _ffmpegDecode.StandardOutput.BaseStream;
            using Stream? outputStream = shouldCreateVideo
                ? _ffmpegEncode.StandardInput.BaseStream
                : default!;

            var frame = new SKBitmap(_videoTargetWidth, _videoTargetHeight, SKColorType.Bgra8888, SKAlphaType.Opaque);

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
                        {
                            OnVideoEnd?.Invoke();
                            return;
                        }

                        bytesRead += read;
                    }

                    // Fill frame with pixels from ffmpeg
                    fixed (byte* ptr = buffer)
                    {
                        frame.SetPixels((nint)ptr);
                    }

                    // Let user process the frame...
                    OnFrameReady?.Invoke(frame, frameIndex);

                    // Encode frame back to video?
                    if (shouldCreateVideo)
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

                if (shouldCreateVideo)
                    _ffmpegEncode.WaitForExit();
            }
        }

        private static (int width, int height) CalculateProportionalResize(Metadata metadata, VideoOptions options)
        {
            int originalWidth = metadata.Width;
            int originalHeight = metadata.Height;
            int targetWidth = options.Width;
            int targetHeight = options.Height;

            targetWidth = targetWidth == -2 || options.Width > 0
                ? options.Width
                : originalWidth;

            targetHeight = targetHeight == -2 || options.Height > 0
                ? options.Height
                : originalHeight;

            if (targetWidth == -2 && targetHeight == -2)
                throw new YoloDotNetVideoException("Both with and height cant be -2.");

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
            var targetFps = _videoTargetfps;

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
                totalFrames = (int)Math.Floor(targetFps * metadata.Duration);
            }
            else
            {
                var fps = metadata.FrameRateDenominator > 1000
                    ? targetFps * 1000 / metadata.FrameRateDenominator
                    : targetFps;

                totalFrames = (int)Math.Round(fps * metadata.Duration);
            }

            // If a custom frame interval is set to run inference on every Nth frame, recalculate total frames.
            if (_videoOptions.FrameInterval > 0)
            {
                totalFrames = (long)Math.Ceiling(totalFrames / (float)_videoOptions.FrameInterval);
            }

            return totalFrames - 1; // Make sure to keep total-frames count zero-indexed.
        }

        private static bool EnsureToolIsInstalled(string fileName)
        {
            try
            {
                using var process = Processor.Create(fileName, ["-version"]);
                process.Start();
                process.WaitForExit(2000); // Timeout after 2 seconds

                return process.ExitCode == 0;
            }
            catch (Win32Exception ex)
            {
                // Common on Windows when tool is not found
                throw new YoloDotNetToolException($"Required '{fileName}' is not installed or not in PATH.", ex);
            }
            catch (Exception ex)
            {
                // Wrap any other issues into a clear higher-level message
                throw new YoloDotNetVideoException($"Failed to verify '{fileName}': {ex.Message}", ex);
            }
        }

        public static List<string> GetVideoDevicesOnSystem()
        {
            EnsureToolIsInstalled(FFMPEG);

            // Detect the platform(Windows or Linux)
            var isLinux = SystemPlatform.GetOS() == Platform.Linux;

            // Set input format and regex pattern based on platform
            string format = isLinux ? "v4l2" : "dshow";
            string pattern = isLinux
                ? @"/dev/video\d+:\s+[^\n]+"        // Linux: Capture both device path and name
                : @"[^""]+(?=""\s+\(video\))";      // Windows: Capture device name in quotes only

            try
            {
                using var ffmpeg = Processor.Create(FFMPEG, [
                    "-f",               format,
                    "-list_devices",    "true",
                    "-i",               "dummy"]);

                ffmpeg.Start();
                string ffmpegOutput = ffmpeg.StandardError.ReadToEnd();

                ffmpeg.WaitForExit(2000);

                var devices = Regex.Matches(ffmpegOutput, pattern)
                    .Select(x => x.Value)
                    .ToList();

                return devices;
            }
            catch (Win32Exception ex)
            {
                throw new YoloDotNetToolException($"FFmpeg is not installed or not in PATH.", ex);
            }
            catch (Exception ex)
            {
                throw new YoloDotNetVideoException($"Failed to get devices: {ex.Message}", ex);
            }

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
