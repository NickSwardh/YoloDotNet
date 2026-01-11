// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Exceptions
{
    public class YoloDotNetModelMismatchException : Exception
    {
        public YoloDotNetModelMismatchException(string message)
            : base(message)
        {
        }

        public YoloDotNetModelMismatchException(string message, Exception innerException)
            : base(message, innerException)
        {
        }
    }
}
