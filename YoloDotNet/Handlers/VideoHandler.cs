using System.Globalization;
using System.Text.RegularExpressions;
using YoloDotNet.Enums;
using YoloDotNet.Extensions;
using YoloDotNet.Models;

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
        private List<string> _output = new();

        public VideoOptions _videoOptions;
        private readonly bool _useCuda;

        public VideoMetaData VideoMetaData { get; private set; } = new();
        public VideoAction CurrentPipelineStep;

        /// <summary>
        /// Initializes a new VindeoHandler instance with the specified video options and CUDA flag.
        /// </summary>
        /// <param name="options">Options for configuring video processing.</param>
        /// <param name="useCuda">Flag indicating whether to use CUDA for video processing.</param>
        public VideoHandler(VideoOptions options, bool useCuda)
        {
            _handler = new ProcessHandler();
            _handler.ProcessStopped += OnProcessCompleteEvent;
            _handler.DataReceived += OnDataReceivedEvent;
            _pipeline = BuildPipeline();
            _videoOptions = options;
            _useCuda = useCuda;
        }

        /// <summary>
        /// Pipeline for processing video
        /// </summary>
        private Dictionary<VideoAction, Action> BuildPipeline()
        {
            return new Dictionary<VideoAction, Action>
            {
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
        public void ProcessVideoPipeline(VideoAction step = VideoAction.GetMetaData)
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
            var data = (string)sender!;
            _output.Add(data);

            Task.Run(() => ProgressAsync(data, VideoMetaData.Frames).ConfigureAwait(false));
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

            if (match.Success)
            {
                var currentFrame = double.Parse(match.Value);
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
                FramesExtractedEvent?.Invoke(VideoMetaData, EventArgs.Empty);
            }
            else
            {
                ProcessVideoPipeline(CurrentPipelineStep + 1);
            }
        }

        /// <summary>
        /// Retrieves information about the specified video file using ffprobe.
        /// </summary>
        /// <exception cref="FileNotFoundException">Thrown if the specified video file is not found.</exception>
        private void GetVideoInfo()
        {
            if (File.Exists(_videoOptions.VideoFile) is false)
                throw new FileNotFoundException($"Could not find: {_videoOptions.VideoFile}");

            Execute($@"-select_streams v:0 -count_frames -show_entries stream=nb_read_frames -i ""{_videoOptions.VideoFile}""", Executable.ffprobe);
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

            var matches = Regex.Matches(metadata, @"(?<=frames=)(?<Frames>\d+)(?=\r\n).*?(?<=Duration: )(?<Duration>\d+:\d+:\d+.\d+).*?(?<=bitrate: )(?<Bitrate>\d+).*?(?<=, )(?<Width>\d+)x(?<Height>\d+).*?(?<FPS>\d+(?:\.\d+)?)(?=\s+fps)", RegexOptions.Singleline);

            if (matches.Count == 0)
                throw new ArgumentException($"Error extracting metadata: {metadata}", nameof(matches));

            VideoMetaData = matches[0].Groups.BuildMetaData(_videoOptions);
            OnProcessCompleteEvent(null, null!);
        }

        /// <summary>
        /// Execute command to extrac audio from video
        /// </summary>
        private void ExtractAudio()
            => Execute($@"-hwaccel auto -i ""{VideoMetaData.VideoFile}"" ""{VideoMetaData.TempFolder.FullName}\audio.mp3"" -y -hide_banner");

        /// <summary>
        /// Execute commant to extract all frames from video
        /// </summary>
        private void ExtractFrames()
        {
            var (w, h) = (VideoMetaData.Width, VideoMetaData.Height);
            Execute($@"-hwaccel auto -i ""{VideoMetaData.VideoFile}"" -vf ""fps={VideoMetaData.FPS},scale={w}:{h}"" ""{VideoMetaData.TempFolder.FullName}\%d.png"" -hide_banner");
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
            var fps = VideoMetaData.FPS.ToString("#.###", CultureInfo.InvariantCulture);

            var folder = VideoMetaData.TempFolder.FullName;
            var audio = _videoOptions.KeepAudio && File.Exists(folder + @"\audio.mp3")
                ? $@"-i ""{folder}\audio.mp3"""
                : "";

            var cv = " libx264";
            var cuda = "";

            if (_useCuda)
            {
                cuda = "-hwaccel_output_format cuda";
                cv = "h264_nvenc";
            }

            Execute($@"-hwaccel auto {cuda} -framerate {fps} -i ""{folder}\%d.png"" {audio} -c:v {cv} -pix_fmt yuv420p ""{VideoMetaData.OutputFolder.FullName}\output.mp4"" -y -hide_banner");
        }

        /// <summary>
        /// Get all extracted frames from temporary folder
        /// </summary>
        public string[] GetExtractedFrames()
            => Directory.GetFiles(VideoMetaData.TempFolder.FullName, "*.png")
                .OrderBy(x => int.Parse(Path.GetFileNameWithoutExtension(x))).ToArray();

        /// <summary>
        /// Execute process
        /// </summary>
        /// <param name="cmd">Commands</param>
        /// <param name="exe">Executable</param>
        private void Execute(string cmd, Executable exe = Executable.ffmpeg)
            => _handler.RunProcess(exe.ToString(), cmd);

        /// <summary>
        /// Regex for matching current frame in progress from ffmpeg stream output
        /// </summary>
        /// <returns></returns>
        [GeneratedRegex("(?<=frame=\\s+)\\d+")]

        private static partial Regex FrameNumberRegex();
        public void Dispose()
        {
            if (_videoOptions.KeepFrames is false)
                VideoMetaData.TempFolder.Delete(true);

            _handler.ProcessStopped -= OnProcessCompleteEvent;
            _handler.DataReceived -= OnDataReceivedEvent;
            _output.Clear();
            GC.SuppressFinalize(this);
        }
    }
}
