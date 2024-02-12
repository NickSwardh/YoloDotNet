namespace YoloDotNet.VideoHandler
{
    public partial class VideoHandler : IDisposable
    {
        public event EventHandler FramesExtractedEvent = delegate { };
        public event EventHandler VideoCompleteEvent = delegate { };
        public event EventHandler ProgressEvent = delegate { };
        public event EventHandler StatusChangeEvent = delegate { };

        private readonly ProcessHandler _handler;
        private readonly Dictionary<VideoAction, Action> _pipeline;
        private List<string> _output = [];

        public VideoSettings _videoSettings;
        private readonly bool _useCuda;

        private const string TEMP_VIDEO_FILENAME = "temp.mp4";
        private const string TEMP_AUDIO_FILENAME = "audio.mkv";
        private const string OUTPUT_FILEMAME = "output.mp4";
        private const string FRAME_FORMAT = "png";

        public VideoAction CurrentPipelineStep;

        /// <summary>
        /// Initializes a new VideoHandler instance with the specified video options and CUDA flag.
        /// </summary>
        /// <param name="options">Options for configuring video processing.</param>
        /// <param name="useCuda">Flag indicating whether to use CUDA for video processing.</param>
        public VideoHandler(VideoOptions options, bool useCuda)
        {
            _handler = new ProcessHandler();
            _videoSettings = (VideoSettings)options;
            _pipeline = BuildPipeline();
            _handler.ProcessStopped += OnProcessCompleteEvent;
            _handler.DataReceived += OnDataReceivedEvent;
            _useCuda = useCuda;
        }

        /// <summary>
        /// Pipeline for processing video
        /// </summary>
        private Dictionary<VideoAction, Action> BuildPipeline()
        {
            return new Dictionary<VideoAction, Action>
            {
                { VideoAction.PreProcess, PreProcess },
                { VideoAction.GetMetaData, GetVideoInfo },
                { VideoAction.ExtractMetaData, ExtractMetaData },
                { VideoAction.ExtractAudio, ExtractAudio },
                { VideoAction.ExtractFrames, ExtractFrames },
                { VideoAction.ProcessFrames, ProcessFrames },
                { VideoAction.CompileFrames, CompileFrames }
            };
        }

        /// <summary>
        /// Process video
        /// </summary>
        public void ProcessVideoPipeline(VideoAction step = VideoAction.PreProcess)
        {
            CurrentPipelineStep = step;
            StatusChangeEvent?.Invoke(step.ToString(), EventArgs.Empty);
            _pipeline[step].Invoke();
        }

        /// <summary>
        /// Data received from process output stream
        /// </summary>
        private void OnDataReceivedEvent(object? sender, EventArgs e)
        {
            if (CurrentPipelineStep <= VideoAction.PreProcess)
                return;

            var data = (string)sender!;

            if (CurrentPipelineStep <= VideoAction.GetMetaData)
                _output.Add(data);

            Task.Run(() => ProgressAsync(data, _videoSettings.TotalFrames).ConfigureAwait(false));
        }

        /// <summary>
        /// Progress counter that triggers a progress event
        /// </summary>
        /// <param name="text">The input text containing information about the current progress.</param>
        /// <param name="frames">The total number of frames for calculating progress.</param>
        private async Task ProgressAsync(string text, int frames)
        {
            await Task.Yield();

            var match = FrameNumberRegex().Match(text);

            if (match.Success && double.TryParse(match.Value, out double currentFrame))
            {
                //var currentFrame = double.Parse(match.Value);
                var progressPercent = (int)(currentFrame / frames * 100);
                ProgressEvent.Invoke(progressPercent, EventArgs.Empty);
            }
        }

        /// <summary>
        /// Event handler for the completion of a video processing step.
        /// Triggers appropriate events based on the current pipeline step.
        /// </summary>
        /// <param name="sender">The object that triggered the event.</param>
        /// <param name="e">Event arguments.</param>
        private void OnProcessCompleteEvent(object? sender, EventArgs e)
        {
            if (CurrentPipelineStep == VideoAction.CompileFrames)
            {
                VideoCompleteEvent?.Invoke(null, EventArgs.Empty);
            }
            else if (CurrentPipelineStep == VideoAction.ProcessFrames)
            {
                FramesExtractedEvent?.Invoke(_videoSettings, EventArgs.Empty);
            }
            else
            {
                ProcessVideoPipeline(CurrentPipelineStep + 1);
            }
        }

        /// <summary>
        /// Pre-process video file by creating a temporary file with video stream only, in order to get actual duration of video and other video info
        /// </summary>
        private void PreProcess()
        {
            VideoExtension.CreateOutputFolder(_videoSettings.TempFolder, true);

            var tempFile = Path.Combine(_videoSettings.TempFolder, TEMP_VIDEO_FILENAME);
            Execute($@"-i ""{_videoSettings.VideoFile}"" -map 0:v:0 -vsync vfr -an -cq 51 -preset ultrafast -c copy ""{tempFile}""");
            Thread.Sleep(100);
        }

        /// <summary>
        /// Retrieves information about the specified video file using ffprobe.
        /// </summary>
        private void GetVideoInfo()
        {
            var tempVideo = Path.Combine(_videoSettings.TempFolder, TEMP_VIDEO_FILENAME);
            Execute($@"-select_streams v:0 -show_entries stream=r_frame_rate,duration ""{tempVideo}""", Executable.ffprobe);
        }

        /// <summary>
        /// Extracts metadata from the output generated during video processing.
        /// Parses information such as frame count, duration, bitrate, resolution, and FPS.
        /// Triggers the OnProcessCompleteEvent upon successful metadata extraction.
        /// </summary>
        /// <exception cref="ArgumentException">Thrown if metadata extraction fails.</exception>
        private void ExtractMetaData()
        {
            var metadata = string.Join("\r\n", _output);
            var matches = VideoMetadataRegex().Matches(metadata);

            if (matches.Count == 0)
                throw new ArgumentException($"Error extracting metadata from video.", nameof(matches));

            // Update settings
            matches[0].Groups.ParseVideoMetaData(_videoSettings);

            // Delete temporary processed file
            Path.Combine(_videoSettings.TempFolder, TEMP_VIDEO_FILENAME).DeleteFile();

            OnProcessCompleteEvent(null, null!);
        }

        /// <summary>
        /// Execute command to extrac audio from video
        /// </summary>
        private void ExtractAudio()
            => Execute($@"-i ""{_videoSettings.VideoFile}"" -vn -c:a copy ""{Path.Combine(_videoSettings.TempFolder, TEMP_AUDIO_FILENAME)}""");

        /// <summary>
        /// Execute commant to extract all frames from video
        /// </summary>
        private void ExtractFrames()
        {
            var (w, h) = (_videoSettings.Width, _videoSettings.Height);
            Execute($@"-i ""{_videoSettings.VideoFile}"" -vsync vfr -vf ""fps={FormatedFps},scale={w}:{h}"" ""{Path.Combine(_videoSettings.TempFolder, $"%d.{FRAME_FORMAT}")}""");
        }

        /// <summary>
        /// Trigger event to signal that inference on extracted frames are ready.
        /// </summary>
        private void ProcessFrames()
            => OnProcessCompleteEvent(null, null!);

        /// <summary>
        /// Execute command to compile all frames to video
        /// </summary>
        private void CompileFrames()
        {
            var tempFolder = _videoSettings.TempFolder;
            var audioPath = Path.Combine(tempFolder, TEMP_AUDIO_FILENAME);
            var outputFile = Path.Combine(_videoSettings.OutputFolder, OUTPUT_FILEMAME);
            var outputImage = Path.Combine(tempFolder, $"%d.{FRAME_FORMAT}");

            var audio = _videoSettings.KeepAudio && File.Exists(audioPath)
                ? $@"-i ""{audioPath}"" -c:a copy"
                : "";

            string cv = _useCuda ? "h264_nvenc" : "libx264";
            string cuda = _useCuda ? "-hwaccel_output_format cuda" : "";

            Execute($@"{cuda} -framerate {FormatedFps} -i ""{outputImage}"" {audio} -c:v {cv} -pix_fmt yuv420p -vf setsar=1:1 ""{outputFile}""");
        }

        /// <summary>
        /// Get all extracted frames from temporary folder
        /// </summary>
        public string[] GetExtractedFrames()
            => Directory.GetFiles(_videoSettings.TempFolder, $"*.{FRAME_FORMAT}")
                .OrderBy(x => int.Parse(Path.GetFileNameWithoutExtension(x))).ToArray();

        /// <summary>
        /// String representation of video FPS value
        /// </summary>
        private string FormatedFps
            => _videoSettings.FPS?.ToString("#.###", CultureInfo.InvariantCulture)!;

        /// <summary>
        /// Execute process
        /// </summary>
        /// <param name="cmd">Commands</param>
        /// <param name="exe">Executable</param>
        private void Execute(string cmd, Executable exe = Executable.ffmpeg)
            => _handler.RunProcess(exe, cmd);

        /// <summary>
        /// Regex for matching current frame in progress from ffmpeg stream output
        /// </summary>
        /// <returns></returns>
        [GeneratedRegex(@"(?<=frame=\s*)\d+")]
        private static partial Regex FrameNumberRegex();

        /// <summary>
        /// Regex for matching metadata from ffprobe output
        /// </summary>
        [GeneratedRegex(@"(?<=r_frame_rate=)(?<Frame>\d+)\/(?<Rate>\d+).*?(?<=duration=)(?<Duration>\d+\.\d+).*?(?<=bitrate: )(?<Bitrate>\d+).*?(?<=, )(?<Width>\d+)x(?<Height>\d+).*?(?<FPS>\d+(?:\.\d+)?)(?=\s+fps)", RegexOptions.Singleline)]
        private static partial Regex VideoMetadataRegex();

        public void Dispose()
        {
            if (_videoSettings.KeepFrames is false)
                Directory.Delete(_videoSettings.TempFolder, true);

            _handler.ProcessStopped -= OnProcessCompleteEvent;
            _handler.DataReceived -= OnDataReceivedEvent;
            _output.Clear();
            GC.SuppressFinalize(this);
        }
    }
}
