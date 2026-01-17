// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Exceptions
{
    public class YoloDotNetExecutionProviderException : Exception
    {
        public YoloDotNetExecutionProviderException(string message)
            : base(message)
        {
        }

        public YoloDotNetExecutionProviderException(string message, Exception innerException)
            : base(message, innerException)
        {
        }
    }
}
