namespace YoloDotNet.Services
{
    public class Processor
    {
        public static Process Create(string process, string[] arguments)
            => new()
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = process,
                    Arguments = string.Join(' ', arguments),
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    RedirectStandardInput = true,
                    CreateNoWindow = true
                },

                EnableRaisingEvents = true,
            };
    }
}
