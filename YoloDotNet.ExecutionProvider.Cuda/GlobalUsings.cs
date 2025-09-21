// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

global using System.Data;
global using System.Runtime.InteropServices;

global using YoloDotNet.Core;
global using YoloDotNet.Models;
global using YoloDotNet.Exceptions;
global using YoloDotNet.Models.Interfaces;
global using YoloDotNet.ExecutionProvider.Cuda.TensorRT;

global using Microsoft.ML.OnnxRuntime;
global using Microsoft.ML.OnnxRuntime.Tensors;