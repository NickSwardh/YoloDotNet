// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Exceptions
{
    public class YoloDotNetModelException : ArgumentException
    {
        public YoloDotNetModelException(string message)
            : base(message)
        {
        }

        public YoloDotNetModelException(string message, string paramName)
            : base(message, paramName)
        {
        }

        public YoloDotNetModelException(string message, Exception innerException)
            : base(message, innerException)
        {
        }
    }
}
