// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Exceptions
{
    public class YoloDotNetUnsupportedProviderException : Exception
    {
        public YoloDotNetUnsupportedProviderException(string message)
            : base(message)
        {
        }

        public YoloDotNetUnsupportedProviderException(string message, Exception innerException)
            : base(message, innerException)
        {
        }
    }
}
