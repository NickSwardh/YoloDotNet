// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Attributes
{
    /// <summary>
    /// Attribute to specify the encoder name associated with an enum value.
    /// </summary>
    /// <param name="name"></param>
    [AttributeUsage(AttributeTargets.Field)]
    public sealed class EncoderNameAttribute(string name) : Attribute
    {
        public string Name { get; } = name;
    }
}
