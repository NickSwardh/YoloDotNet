// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.ExecutionProvider.OpenVino
{
    public class OpenVino
    {
        public string DeviceType { get; set; } = default!;
        public Precision Precision { get; set; }
        public int Threads { get; set; } = 8; // Open Vino default to 8 threads
        public int Streams { get; set; } = 1; // Open Vino default to 1 stream
        public string CachePath { get; set; } = default!;
        public ModelPriority ModelPriority { get; set; }
    }
}
