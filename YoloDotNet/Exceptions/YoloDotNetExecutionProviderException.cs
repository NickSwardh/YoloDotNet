// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
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
