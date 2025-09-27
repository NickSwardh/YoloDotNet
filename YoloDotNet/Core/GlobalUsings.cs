// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

global using System.Data;
global using System.Buffers;
global using System.Diagnostics;
global using System.Globalization;
global using System.ComponentModel;
global using System.Collections.Concurrent;

global using System.Text.Json;
global using System.Text.RegularExpressions;
global using System.Text.Json.Serialization;

global using System.Runtime.Intrinsics;
global using System.Runtime.Serialization;
global using System.Runtime.Intrinsics.X86;
global using System.Runtime.InteropServices;
global using System.Runtime.CompilerServices;

global using SkiaSharp;

global using YoloDotNet.Core;
global using YoloDotNet.Enums;
global using YoloDotNet.Video;
global using YoloDotNet.Utils;
global using YoloDotNet.Models;
global using YoloDotNet.Handlers;
global using YoloDotNet.Trackers;
global using YoloDotNet.Attributes;
global using YoloDotNet.Extensions;
global using YoloDotNet.Exceptions;
global using YoloDotNet.Configuration;
global using YoloDotNet.Video.Services;
global using YoloDotNet.Models.Interfaces;

global using YoloDotNet.Modules.V5U;
global using YoloDotNet.Modules.V8;
global using YoloDotNet.Modules.V8E;
global using YoloDotNet.Modules.V9;
global using YoloDotNet.Modules.V10;
global using YoloDotNet.Modules.V11;
global using YoloDotNet.Modules.V11E;
global using YoloDotNet.Modules.V12;
global using YoloDotNet.Modules.WorldV2;
global using YoloDotNet.Modules.Interfaces;
