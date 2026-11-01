// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.ExecutionProvider.Cuda
{
    /// <summary>CUDA runtime functions used only by the opt-in graph path, matching ORT's CUDA 12 dependency.</summary>
    internal static class CudaRuntime
    {
        // Keep the module loaded for the lifetime of these delegates.
        private static readonly nint _library = NativeLibrary.Load(
            OperatingSystem.IsWindows() ? "cudart64_12.dll" : "libcudart.so.12");

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate int SetDeviceDelegate(int device);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate int CreateStreamDelegate(out nint stream, uint flags);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate int StreamDelegate(nint stream);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate int CopyDelegate(nint destination, nint source, nuint bytes, int kind, nint stream);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate nint ErrorDelegate(int error);

        private static readonly SetDeviceDelegate _setDeviceNative = Load<SetDeviceDelegate>("cudaSetDevice");
        private static readonly CreateStreamDelegate _createStreamNative = Load<CreateStreamDelegate>("cudaStreamCreateWithFlags");
        private static readonly StreamDelegate _synchronizeNative = Load<StreamDelegate>("cudaStreamSynchronize");
        private static readonly StreamDelegate _destroyStreamNative = Load<StreamDelegate>("cudaStreamDestroy");
        private static readonly CopyDelegate _copyNative = Load<CopyDelegate>("cudaMemcpyAsync");
        private static readonly ErrorDelegate _errorNative = Load<ErrorDelegate>("cudaGetErrorString");

        private static T Load<T>(string name) where T : Delegate
            => Marshal.GetDelegateForFunctionPointer<T>(NativeLibrary.GetExport(_library, name));

        internal static void SetDevice(int device) => Check(_setDeviceNative(device));

        internal static nint CreateStream()
        {
            Check(_createStreamNative(out var stream, 1)); // cudaStreamNonBlocking
            return stream;
        }

        internal static void DestroyStream(nint stream) => _destroyStreamNative(stream);

        internal static void Copy(nint destination, nint source, uint bytes, bool upload, nint stream)
        {
            Check(_copyNative(destination, source, bytes, upload ? 1 : 2, stream));
            // GC pinning is not CUDA page-locking. A host upload may return after staging, before DMA
            // finishes. Wait for our copy stream before launching the graph or releasing the host pin.
            Check(_synchronizeNative(stream));
        }

        private static void Check(int error)
        {
            if (error != 0)
                throw new YoloDotNetExecutionProviderException($"CUDA error {error}: {Marshal.PtrToStringAnsi(_errorNative(error))}");
        }
    }
}
