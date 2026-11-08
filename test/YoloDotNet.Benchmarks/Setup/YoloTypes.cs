// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Benchmarks.Setup
{
    public enum YoloType
    {
        V5u_Cls_CPU,
        V5u_Cls_GPU,
        V5u_Cls_TRT32,
        V5u_Cls_TRT16,
        V5u_Cls_TRT8,

        V5u_Obj_CPU,
        V5u_Obj_GPU,
        V5u_Obj_TRT32,
        V5u_Obj_TRT16,
        V5u_Obj_TRT8,

        V5u_Obb_CPU,
        V5u_Obb_GPU,
        V5u_Obb_TRT32,
        V5u_Obb_TRT16,
        V5u_Obb_TRT8,

        V5u_Pos_CPU,
        V5u_Pos_GPU,
        V5u_Pos_TRT32,
        V5u_Pos_TRT16,
        V5u_Pos_TRT8,

        V5u_Seg_CPU,
        V5u_Seg_GPU,
        V5u_Seg_TRT32,
        V5u_Seg_TRT16,
        V5u_Seg_TRT8,

        //

        V8_Cls_CPU,
        V8_Cls_GPU,
        V8_Cls_TRT32,
        V8_Cls_TRT16,
        V8_Cls_TRT8,

        V8_Obj_CPU,
        V8_Obj_GPU,
        V8_Obj_TRT32,
        V8_Obj_TRT16,
        V8_Obj_TRT8,

        V8_Obb_CPU,
        V8_Obb_GPU,
        V8_Obb_TRT32,
        V8_Obb_TRT16,
        V8_Obb_TRT8,

        V8_Pos_CPU,
        V8_Pos_GPU,
        V8_Pos_TRT32,
        V8_Pos_TRT16,
        V8_Pos_TRT8,

        V8_Seg_CPU,
        V8_Seg_GPU,
        V8_Seg_TRT32,
        V8_Seg_TRT16,
        V8_Seg_TRT8,

        //

        V9_Cls_CPU,
        V9_Cls_GPU,
        V9_Cls_TRT32,
        V9_Cls_TRT16,
        V9_Cls_TRT8,

        V9_Obj_CPU,
        V9_Obj_GPU,
        V9_Obj_TRT32,
        V9_Obj_TRT16,
        V9_Obj_TRT8,

        V9_Obb_CPU,
        V9_Obb_GPU,
        V9_Obb_TRT32,
        V9_Obb_TRT16,
        V9_Obb_TRT8,

        V9_Pos_CPU,
        V9_Pos_GPU,
        V9_Pos_TRT32,
        V9_Pos_TRT16,
        V9_Pos_TRT8,

        V9_Seg_CPU,
        V9_Seg_GPU,
        V9_Seg_TRT32,
        V9_Seg_TRT16,
        V9_Seg_TRT8,

        //

        V10_Cls_CPU,
        V10_Cls_GPU,
        V10_Cls_TRT32,
        V10_Cls_TRT16,
        V10_Cls_TRT8,

        V10_Obj_CPU,
        V10_Obj_GPU,
        V10_Obj_TRT32,
        V10_Obj_TRT16,
        V10_Obj_TRT8,

        V10_Obb_CPU,
        V10_Obb_GPU,
        V10_Obb_TRT32,
        V10_Obb_TRT16,
        V10_Obb_TRT8,

        V10_Pos_CPU,
        V10_Pos_GPU,
        V10_Pos_TRT32,
        V10_Pos_TRT16,
        V10_Pos_TRT8,

        V10_Seg_CPU,
        V10_Seg_GPU,
        V10_Seg_TRT32,
        V10_Seg_TRT16,
        V10_Seg_TRT8,

        //

        V11_Cls_CPU,
        V11_Cls_GPU,
        V11_Cls_TRT32,
        V11_Cls_TRT16,
        V11_Cls_TRT8,

        V11_Obj_CPU,
        V11_Obj_GPU,
        V11_Obj_TRT32,
        V11_Obj_TRT16,
        V11_Obj_TRT8,

        V11_Obb_CPU,
        V11_Obb_GPU,
        V11_Obb_TRT32,
        V11_Obb_TRT16,
        V11_Obb_TRT8,

        V11_Pos_CPU,
        V11_Pos_GPU,
        V11_Pos_TRT32,
        V11_Pos_TRT16,
        V11_Pos_TRT8,

        V11_Seg_CPU,
        V11_Seg_GPU,
        V11_Seg_TRT32,
        V11_Seg_TRT16,
        V11_Seg_TRT8,

        //

        V12_Cls_CPU,
        V12_Cls_GPU,
        V12_Cls_TRT32,
        V12_Cls_TRT16,
        V12_Cls_TRT8,

        V12_Obj_CPU,
        V12_Obj_GPU,
        V12_Obj_TRT32,
        V12_Obj_TRT16,
        V12_Obj_TRT8,

        V12_Obb_CPU,
        V12_Obb_GPU,
        V12_Obb_TRT32,
        V12_Obb_TRT16,
        V12_Obb_TRT8,

        V12_Pos_CPU,
        V12_Pos_GPU,
        V12_Pos_TRT32,
        V12_Pos_TRT16,
        V12_Pos_TRT8,

        V12_Seg_CPU,
        V12_Seg_GPU,
        V12_Seg_TRT32,
        V12_Seg_TRT16,
        V12_Seg_TRT8,

        //

        V26_Cls_CPU,
        V26_Cls_GPU,
        V26_Cls_TRT32,
        V26_Cls_TRT16,
        V26_Cls_TRT8,

        V26_Obj_CPU,
        V26_Obj_GPU,
        V26_Obj_TRT32,
        V26_Obj_TRT16,
        V26_Obj_TRT8,

        V26_Obb_CPU,
        V26_Obb_GPU,
        V26_Obb_TRT32,
        V26_Obb_TRT16,
        V26_Obb_TRT8,

        V26_Pos_CPU,
        V26_Pos_GPU,
        V26_Pos_TRT32,
        V26_Pos_TRT16,
        V26_Pos_TRT8,

        V26_Seg_CPU,
        V26_Seg_GPU,
        V26_Seg_TRT32,
        V26_Seg_TRT16,
        V26_Seg_TRT8,

        //

        RTDETR_Obj_CPU,
        RTDETR_Obj_GPU,
        RTDETR_Obj_TRT32,
        RTDETR_Obj_TRT16,
        RTDETR_Obj_TRT8,

        // GPU with CUDA Graphs enabled

        V5u_Obj_GPUGraph,

        V8_Cls_GPUGraph,
        V8_Obj_GPUGraph,
        V8_Obb_GPUGraph,
        V8_Pos_GPUGraph,
        V8_Seg_GPUGraph,

        V9_Obj_GPUGraph,
        V10_Obj_GPUGraph,

        V11_Cls_GPUGraph,
        V11_Obj_GPUGraph,
        V11_Obb_GPUGraph,
        V11_Pos_GPUGraph,
        V11_Seg_GPUGraph,

        V12_Obj_GPUGraph,

        V26_Cls_GPUGraph,
        V26_Obj_GPUGraph,
        V26_Obb_GPUGraph,
        V26_Pos_GPUGraph,
        V26_Seg_GPUGraph,

        // TensorRT with CUDA Graphs enabled

        V5u_Obj_TRT32Graph,
        V5u_Obj_TRT16Graph,

        V8_Cls_TRT32Graph,
        V8_Cls_TRT16Graph,
        V8_Obj_TRT32Graph,
        V8_Obj_TRT16Graph,
        V8_Obb_TRT32Graph,
        V8_Obb_TRT16Graph,
        V8_Pos_TRT32Graph,
        V8_Pos_TRT16Graph,
        V8_Seg_TRT32Graph,
        V8_Seg_TRT16Graph,

        V9_Obj_TRT32Graph,
        V9_Obj_TRT16Graph,
        V10_Obj_TRT32Graph,
        V10_Obj_TRT16Graph,

        V11_Cls_TRT32Graph,
        V11_Cls_TRT16Graph,
        V11_Obj_TRT32Graph,
        V11_Obj_TRT16Graph,
        V11_Obb_TRT32Graph,
        V11_Obb_TRT16Graph,
        V11_Pos_TRT32Graph,
        V11_Pos_TRT16Graph,
        V11_Seg_TRT32Graph,
        V11_Seg_TRT16Graph,

        V12_Obj_TRT32Graph,
        V12_Obj_TRT16Graph,

        V26_Cls_TRT32Graph,
        V26_Cls_TRT16Graph,
        V26_Obj_TRT32Graph,
        V26_Obj_TRT16Graph,
        V26_Obb_TRT32Graph,
        V26_Obb_TRT16Graph,
        V26_Pos_TRT32Graph,
        V26_Pos_TRT16Graph,
        V26_Seg_TRT32Graph,
        V26_Seg_TRT16Graph,
    }
}
