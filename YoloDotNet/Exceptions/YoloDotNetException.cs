// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Exceptions
{
    public class YoloDotNetException : ArgumentException
    {
        public YoloDotNetException(string message)
            : base(message)
        {
        }

        public YoloDotNetException(string message, string paramName)
            : base(message, paramName)
        {
        }

        public YoloDotNetException(string message, Exception innerException)
            : base(message, innerException)
        {
        }
    }
}
