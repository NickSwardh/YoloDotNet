// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Enums
{
    /// <summary>
    /// Common video encoders.
    /// Note: FFmpeg must be built with support for the selected encoder.
    /// </summary>
    public enum VideoEncoder
    {
        #region H.264 / AVC

        /// <summary>
        /// Software H.264 encoder (CPU).
        /// Universally supported, good quality, slower performance.
        /// </summary>
        [EncoderName("libx264")]
        LibX264,

        /// <summary>
        /// NVIDIA NVENC H.264 (GPU).
        /// Fast hardware encoder. Requires NVIDIA GPU and FFmpeg built with NVENC.
        /// </summary>
        [EncoderName("h264_nvenc")]
        H264Nvenc,

        /// <summary>
        /// Intel Quick Sync H.264 (GPU).
        /// Requires Intel iGPU and FFmpeg built with QSV support.
        /// </summary>
        [EncoderName("h264_qsv")]
        H264Qsv,

        /// <summary>
        /// AMD AMF H.264 (GPU).
        /// Requires AMD GPU and FFmpeg built with AMF.
        /// </summary>
        [EncoderName("h264_amf")]
        H264Amf,

        /// <summary>
        /// VAAPI H.264 (Linux only).
        /// Generic hardware encoder for Intel/AMD GPUs.
        /// </summary>
        [EncoderName("h264_vaapi")]
        H264Vaapi,

        /// <summary>
        /// Apple VideoToolbox H.264 (macOS).
        /// Hardware-accelerated encoding on Apple devices.
        /// </summary>
        [EncoderName("h264_videotoolbox")]
        H264VideoToolbox,

        #endregion

        #region H.265 / HEVC

        /// <summary>
        /// Software H.265/HEVC encoder (CPU).
        /// Very efficient compression, but slow compared to hardware.
        /// </summary>
        [EncoderName("libx265")]
        LibX265,

        /// <summary>
        /// NVIDIA NVENC HEVC (GPU).
        /// Requires NVIDIA GPU and FFmpeg with NVENC.
        /// </summary>
        [EncoderName("hevc_nvenc")]
        HevcNvenc,

        /// <summary>
        /// Intel Quick Sync HEVC (GPU).
        /// Requires Intel iGPU and FFmpeg built with QSV support.
        /// </summary>
        [EncoderName("hevc_qsv")]
        HevcQsv,

        /// <summary>
        /// AMD AMF HEVC (GPU).
        /// Requires AMD GPU and FFmpeg built with AMF.
        /// </summary>
        [EncoderName("hevc_amf")]
        HevcAmf,

        /// <summary>
        /// VAAPI HEVC (Linux only).
        /// Generic hardware encoder for Intel/AMD GPUs.
        /// </summary>
        [EncoderName("hevc_vaapi")]
        HevcVaapi,

        /// <summary>
        /// Apple VideoToolbox HEVC (macOS).
        /// Hardware-accelerated HEVC encoding on Apple devices.
        /// </summary>
        [EncoderName("hevc_videotoolbox")]
        HevcVideoToolbox,

        #endregion

        #region AV1

        /// <summary>
        /// Software AV1 encoder (CPU).
        /// Very efficient compression but extremely slow.
        /// </summary>
        [EncoderName("libaom-av1")]
        LibAomAv1,

        /// <summary>
        /// NVIDIA NVENC AV1 (GPU).
        /// Requires NVIDIA Ampere/Ada GPU (RTX 30/40 series) and FFmpeg built with NVENC.
        /// </summary>
        [EncoderName("av1_nvenc")]
        Av1Nvenc,

        /// <summary>
        /// Intel Quick Sync AV1 (GPU).
        /// Requires Intel Arc GPU or recent iGPU, FFmpeg built with QSV.
        /// </summary>
        [EncoderName("av1_qsv")]
        Av1Qsv,

        /// <summary>
        /// VAAPI AV1 (Linux only).
        /// Generic hardware encoder for Intel/AMD GPUs.
        /// </summary>
        [EncoderName("av1_vaapi")]
        Av1Vaapi,

        /// <summary>
        /// Apple VideoToolbox AV1 (macOS 14+).
        /// Hardware-accelerated AV1 encoding on Apple Silicon.
        /// </summary>
        [EncoderName("av1_videotoolbox")]
        Av1VideoToolbox,

        #endregion

        #region ProRes

        /// <summary>
        /// Apple ProRes encoder via VideoToolbox (macOS).
        /// High-quality, intraframe codec used for editing workflows.
        /// </summary>
        [EncoderName("prores_videotoolbox")]
        ProResVideoToolbox

        #endregion
    }
}
