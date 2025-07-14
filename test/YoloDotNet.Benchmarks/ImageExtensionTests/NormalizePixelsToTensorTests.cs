// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2024-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

﻿namespace YoloDotNet.Benchmarks.ImageExtensionTests
{
    //[CPUUsageDiagnoser]
    [MemoryDiagnoser]
    public class NormalizePixelsToTensorTests
    {
        private static readonly string _testImage = SharedConfig.GetTestImage(ImageType.Street);
        private ArrayPool<float> _customSizeFloatPool;

        private float[] _tensorArrayBuffer;
        private int _tensorBufferSize;
        private PinnedMemoryBufferPool _pinnedBufferPool;

        private Yolo _yolo;
        private SKBitmap _image;

        private static IntPtr _imagePointer;

        [GlobalSetup]
        public void GlobalSetup()
        {
            var options = new YoloOptions
            {
                OnnxModel = YoloCreator.Model8_ObjectDetection,
                Cuda = false
            };

            _yolo = new Yolo(options);
            _image = SKBitmap.Decode(_testImage);

            var imageInfo = new SKImageInfo(_yolo.OnnxModel.Input.Width, _yolo.OnnxModel.Input.Height, SKColorType.Rgb888x, SKAlphaType.Opaque);

            _pinnedBufferPool = new PinnedMemoryBufferPool(imageInfo);
            var pinnedBuffer = _pinnedBufferPool.Rent();

            try
            {
                _ = _image.ResizeImageProportional(options.SamplingOptions, pinnedBuffer);
                _imagePointer = pinnedBuffer.Pointer;
            }
            finally
            {
                _pinnedBufferPool.Return(pinnedBuffer);
            }

            _tensorBufferSize = _yolo.OnnxModel.Input.BatchSize * _yolo.OnnxModel.Input.Channels * _yolo.OnnxModel.Input.Width * _yolo.OnnxModel.Input.Height;
            _customSizeFloatPool = ArrayPool<float>.Create(_tensorBufferSize + 1, 10);
            _tensorArrayBuffer = _customSizeFloatPool.Rent(_tensorBufferSize);
        }

        [GlobalCleanup]
        public void GlobalCleanup()
        {
            _customSizeFloatPool.Return(_tensorArrayBuffer, true);

            _image?.Dispose();
            _yolo?.Dispose();
            _pinnedBufferPool?.Dispose();
        }

        [Benchmark]
        public void NormalizePixelsToTensor()
            => _imagePointer.NormalizePixelsToTensor(_yolo.OnnxModel.InputShape, _tensorBufferSize, _tensorArrayBuffer);
    }
}
