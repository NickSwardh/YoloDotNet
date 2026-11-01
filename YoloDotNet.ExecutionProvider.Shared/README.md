# YoloDotNet.ExecutionProvider.Shared

This is a **shared project** (`.shproj`): it holds source code used by all execution providers, but it is not built or
published on its own. It produces no DLL and no NuGet package.

Each execution provider imports it with a *Shared Project Reference*, and its files are compiled into that provider's
own assembly, against that provider's ONNX Runtime package (`Microsoft.ML.OnnxRuntime`, `.Gpu`, `.DirectML` or
`Intel.ML.OnnxRuntime.OpenVino`). This is how the providers share code that uses ONNX Runtime types, while the core
**YoloDotNet** library itself does not depend on ONNX Runtime.

## Contents

- `Extensions/InferenceSessionExtensions.cs`: reads the model information (input and output shapes, metadata, data
  type) from an ONNX Runtime session and builds the `OnnxModel`.

## Adding code

- Add files to this project; every provider that references it compiles them automatically.
- Types should be `internal`: they become part of each provider, not of a public API.
- Use only namespaces from the providers' `GlobalUsings.cs`; every provider needs the namespaces the shared code uses.
