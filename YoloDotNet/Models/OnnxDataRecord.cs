// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Models
{
    /// <summary>
    /// Record to hold ONNX model data from execution providers.
    /// </summary>
    /// <param name="MetaData"></param>
    /// <param name="InputName"></param>
    /// <param name="OutputNames"></param>
    /// <param name="InputShape"></param>
    /// <param name="OutputShapes"></param>
    /// <param name="Labels"></param>
    public record OnnxDataRecord(
        Dictionary<string, string> MetaData,
        string InputName,
        string[] OutputNames,
        int[] InputShape,
        int[][] OutputShapes,
        string Labels
    );
}