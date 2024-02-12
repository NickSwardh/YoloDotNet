namespace YoloDotNet.VideoHandler
{
    /// <summary>
    /// Handles the execution of external processes, providing events for data reception and process completion.
    /// </summary>
    public class ProcessHandler
    {
        public event EventHandler ProcessStopped = delegate { };
        public event EventHandler DataReceived = delegate { };

        /// <summary>
        /// Start new process
        /// </summary>
        /// <param name="executable"></param>
        /// <param name="arguments"></param>
        public void RunProcess(Executable executable, string arguments)
        {
            using Process process = new()
            {
                StartInfo = new()
                {
                    FileName = executable.ToString(),
                    Arguments = ValidateCommandLineArguments(executable, arguments),
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                }
            };

            process.EnableRaisingEvents = true;  // Enable process exit event
            process.OutputDataReceived += (sender, e) => OnDataReceivedEventHandler(sender, e);
            process.ErrorDataReceived += (sender, e) => OnDataReceivedEventHandler(sender, e);
            process.Exited += (sender, e) => HandleExitEvent(sender, e);

            process.Start();
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
            process.WaitForExit();
        }

        private static string ValidateCommandLineArguments(Executable executable, string arguments)
        {
            //if (arguments is null)
            //    throw new ArgumentNullException(nameof(arguments), "Command-line arguments can't be null.");
            ArgumentNullException.ThrowIfNull(nameof(arguments), "Command-line arguments can't be null.");

            return executable == Executable.ffprobe ? arguments : $@"-hwaccel auto {arguments} -y -hide_banner";
        }

        /// <summary>
        /// Event that will be raised when receiving data
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void OnDataReceivedEventHandler(object sender, DataReceivedEventArgs e)
        {
            if (string.IsNullOrEmpty(e.Data) is false && string.IsNullOrWhiteSpace(e.Data) is false)
                DataReceived.Invoke(e.Data, EventArgs.Empty);
        }

        /// <summary>
        /// Event that will be raised when process stops
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void HandleExitEvent(object? sender, EventArgs e)
        {
            ProcessStopped.Invoke(this, EventArgs.Empty);
        }
    }
}