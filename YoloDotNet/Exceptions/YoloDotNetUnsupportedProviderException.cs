// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 Niklas Swärd
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
