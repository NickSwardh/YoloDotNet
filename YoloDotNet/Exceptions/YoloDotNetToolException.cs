// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Exceptions
{
    public class YoloDotNetToolException : ArgumentException
    {
        public YoloDotNetToolException(string message)
            : base(message)
        {
        }

        public YoloDotNetToolException(string message, string paramName)
            : base(message, paramName)
        {
        }

        public YoloDotNetToolException(string message, Exception innerException)
            : base(message, innerException)
        {
        }
    }
}
