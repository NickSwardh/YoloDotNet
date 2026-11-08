// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.ExecutionProvider.Cuda
{
    /// <summary>
    /// Fixed device allocations for one captured graph. The provider serializes all access, including copies.
    /// Each call downloads into independently owned host buffers so post-processing can overlap the next call.
    /// </summary>
    internal sealed class CudaGraphRunner : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly int _gpuId;
        private readonly List<OutputBuffer> _outputs = [];
        private OrtIoBinding? _binding;
        private OrtMemoryInfo? _memoryInfo;
        private OrtAllocator? _allocator;
        private OrtMemoryAllocation? _input;
        private nint _copyStream;
        private int _inputLength;
        private int _inputElementSize;
        private uint _inputBytes;
        private bool _disposed;

        private sealed record OutputBuffer(OrtMemoryAllocation Memory, int Length, bool IsHalf);

        internal CudaGraphRunner(InferenceSession session, int gpuId)
        {
            _session = session;
            _gpuId = gpuId;
            try
            {
                if (session.InputNames.Count != 1 || session.OutputNames.Count is < 1 or > 2)
                    throw new YoloDotNetExecutionProviderException("CUDA Graphs require one input and one or two outputs.");

                CudaRuntime.SetDevice(gpuId);
                _copyStream = CudaRuntime.CreateStream();
                _memoryInfo = new OrtMemoryInfo("Cuda", OrtAllocatorType.DeviceAllocator, gpuId, OrtMemType.Default);
                _allocator = new OrtAllocator(session, _memoryInfo);
                _binding = session.CreateIoBinding();

                var metadata = session.InputMetadata[session.InputNames[0]];
                _inputLength = TensorLength(metadata);
                _inputElementSize = ElementSize(metadata.ElementDataType);
                _inputBytes = checked((uint)_inputLength * (uint)_inputElementSize);
                _input = _allocator.Allocate(_inputBytes);
                _binding.BindInput(session.InputNames[0], metadata.ElementDataType,
                    Array.ConvertAll(metadata.Dimensions, x => (long)x), _input);

                foreach (var name in session.OutputNames)
                {
                    metadata = session.OutputMetadata[name];
                    int length = TensorLength(metadata);
                    int size = ElementSize(metadata.ElementDataType);
                    var allocation = _allocator.Allocate(checked((uint)length * (uint)size));
                    _outputs.Add(new OutputBuffer(allocation, length, size == sizeof(ushort)));
                    _binding.BindOutput(name, metadata.ElementDataType,
                        Array.ConvertAll(metadata.Dimensions, x => (long)x), allocation);
                }
            }
            catch
            {
                Dispose();
                throw;
            }
        }

        private static int ElementSize(TensorElementType type) => type switch
        {
            TensorElementType.Float => sizeof(float),
            TensorElementType.Float16 => sizeof(ushort),
            _ => throw new YoloDotNetExecutionProviderException("CUDA Graphs support Float32 and Float16 tensors only.")
        };

        private static int TensorLength(NodeMetadata metadata)
        {
            if (!metadata.IsTensor || metadata.Dimensions.Length == 0 || metadata.Dimensions.Any(x => x <= 0))
                throw new YoloDotNetExecutionProviderException("CUDA Graphs require fixed, positive input and output dimensions.");

            return metadata.Dimensions.Aggregate(1, (length, dimension) => checked(length * dimension));
        }

        internal unsafe InferenceResult Run<T>(T[] pixels, RunOptions options) where T : unmanaged
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            ArgumentNullException.ThrowIfNull(pixels);
            if (sizeof(T) != _inputElementSize || pixels.Length < _inputLength)
                throw new ArgumentException("The input buffer does not match the model's tensor type or size.", nameof(pixels));

            // Callers may arrive on different threads, each with its own current CUDA device.
            CudaRuntime.SetDevice(_gpuId);
            fixed (T* source = pixels)
                CudaRuntime.Copy(_input!.DangerousGetHandle(), (nint)source, _inputBytes, true, _copyStream);

            try
            {
                _session.RunWithBinding(options, _binding!);
            }
            catch (OnnxRuntimeException ex)
            {
                throw new YoloDotNetExecutionProviderException(
                    "This model does not support CUDA Graphs: inference failed while the graph was captured or replayed. " +
                    $"Set enableCudaGraph to false to run it without CUDA Graphs. ONNX Runtime said: {ex.Message}", ex);
            }
            _binding!.SynchronizeBoundOutputs();

            float[]? first = null;
            float[]? second = null;
            try
            {
                first = Download(_outputs[0]);
                if (_outputs.Count == 2)
                    second = Download(_outputs[1]);

                return new InferenceResult(null, first.AsSpan(0, _outputs[0].Length),
                    second is null ? ReadOnlySpan<float>.Empty : second.AsSpan(0, _outputs[1].Length), first, second);
            }
            catch
            {
                if (first is not null) ArrayPool<float>.Shared.Return(first);
                if (second is not null) ArrayPool<float>.Shared.Return(second);
                throw;
            }
        }

        private unsafe float[] Download(OutputBuffer output)
        {
            var values = ArrayPool<float>.Shared.Rent(output.Length);
            try
            {
                if (output.IsHalf)
                {
                    var halves = ArrayPool<Half>.Shared.Rent(output.Length);
                    try
                    {
                        fixed (Half* destination = halves)
                            CudaRuntime.Copy((nint)destination, output.Memory.DangerousGetHandle(),
                                checked((uint)output.Length * sizeof(ushort)), false, _copyStream);
                        ((ReadOnlySpan<Half>)halves.AsSpan(0, output.Length)).ConvertToSingle(values);
                    }
                    finally
                    {
                        ArrayPool<Half>.Shared.Return(halves);
                    }
                }
                else
                {
                    fixed (float* destination = values)
                        CudaRuntime.Copy((nint)destination, output.Memory.DangerousGetHandle(),
                            checked((uint)output.Length * sizeof(float)), false, _copyStream);
                }
                return values;
            }
            catch
            {
                ArrayPool<float>.Shared.Return(values);
                throw;
            }
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _binding?.Dispose();
            foreach (var output in _outputs) output.Memory.Dispose();
            _input?.Dispose();
            _allocator?.Dispose();
            _memoryInfo?.Dispose();
            if (_copyStream != nint.Zero)
            {
                CudaRuntime.SetDevice(_gpuId);
                CudaRuntime.DestroyStream(_copyStream);
            }
        }
    }
}
