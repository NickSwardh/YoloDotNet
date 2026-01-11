// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Models
{
    public record OnnxDataRecord(
        Dictionary<string, long[]> Inputs,
        Dictionary<string, int[]> Outputs,
        Dictionary<string, string> MetaData,
        ModelDataType ModelDataType,
        int InputShapeSize
    );
}