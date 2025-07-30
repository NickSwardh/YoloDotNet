namespace YoloDotNet.Exceptions
{
    public class YoloDotNetVideoException : ArgumentException
    {
        public YoloDotNetVideoException(string message)
            : base(message)
        {
        }

        public YoloDotNetVideoException(string message, string paramName)
            : base(message, paramName)
        {
        }

        public YoloDotNetVideoException(string message, Exception innerException)
            : base(message, innerException)
        {
        }
    }
}
