// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Benchmarks.Setup
{
    public class YoloCreator
    {
        // v5u
        public static string Model5u_Classification => SharedConfig.GetTestModelV5U(ModelType.Classification);
        public static string Model5u_ObjectDetection => SharedConfig.GetTestModelV5U(ModelType.ObjectDetection);
        public static string Model5u_ObbDetection => SharedConfig.GetTestModelV5U(ModelType.ObbDetection);
        public static string Model5u_PoseEstimation => SharedConfig.GetTestModelV5U(ModelType.PoseEstimation);
        public static string Model5u_Segmentation => SharedConfig.GetTestModelV5U(ModelType.Segmentation);

        // v8
        public static string Model8_Classification => SharedConfig.GetTestModelV8(ModelType.Classification);
        public static string Model8_ObjectDetection => SharedConfig.GetTestModelV8(ModelType.ObjectDetection);
        public static string Model8_ObbDetection => SharedConfig.GetTestModelV8(ModelType.ObbDetection);
        public static string Model8_PoseEstimation => SharedConfig.GetTestModelV8(ModelType.PoseEstimation);
        public static string Model8_Segmentation => SharedConfig.GetTestModelV8(ModelType.Segmentation);

        // v9
        public static string Model9_Classification => SharedConfig.GetTestModelV9(ModelType.Classification);
        public static string Model9_ObjectDetection => SharedConfig.GetTestModelV9(ModelType.ObjectDetection);
        public static string Model9_ObbDetection => SharedConfig.GetTestModelV9(ModelType.ObbDetection);
        public static string Model9_PoseEstimation => SharedConfig.GetTestModelV9(ModelType.PoseEstimation);
        public static string Model9_Segmentation => SharedConfig.GetTestModelV9(ModelType.Segmentation);

        // v10
        public static string Model10_Classification => SharedConfig.GetTestModelV10(ModelType.Classification);
        public static string Model10_ObjectDetection => SharedConfig.GetTestModelV10(ModelType.ObjectDetection);
        public static string Model10_ObbDetection => SharedConfig.GetTestModelV10(ModelType.ObbDetection);
        public static string Model10_PoseEstimation => SharedConfig.GetTestModelV10(ModelType.PoseEstimation);
        public static string Model10_Segmentation => SharedConfig.GetTestModelV10(ModelType.Segmentation);

        // v11
        public static string Model11_Classification => SharedConfig.GetTestModelV11(ModelType.Classification);
        public static string Model11_ObjectDetection => SharedConfig.GetTestModelV11(ModelType.ObjectDetection);
        public static string Model11_ObbDetection => SharedConfig.GetTestModelV11(ModelType.ObbDetection);
        public static string Model11_PoseEstimation => SharedConfig.GetTestModelV11(ModelType.PoseEstimation);
        public static string Model11_Segmentation => SharedConfig.GetTestModelV11(ModelType.Segmentation);

        // v12
        public static string Model12_Classification => SharedConfig.GetTestModelV12(ModelType.Classification);
        public static string Model12_ObjectDetection => SharedConfig.GetTestModelV12(ModelType.ObjectDetection);
        public static string Model12_ObbDetection => SharedConfig.GetTestModelV12(ModelType.ObbDetection);
        public static string Model12_PoseEstimation => SharedConfig.GetTestModelV12(ModelType.PoseEstimation);
        public static string Model12_Segmentation => SharedConfig.GetTestModelV12(ModelType.Segmentation);

        public static Yolo CreateYolo(YoloType type) => type switch
        {
            YoloType.V5u_Cls_CPU => new Yolo(new YoloOptions { OnnxModel = Model5u_Classification, Cuda = false }),
            YoloType.V5u_Cls_GPU => new Yolo(new YoloOptions { OnnxModel = Model5u_Classification }),
            YoloType.V5u_Obj_CPU => new Yolo(new YoloOptions { OnnxModel = Model5u_ObjectDetection, Cuda = false }),
            YoloType.V5u_Obj_GPU => new Yolo(new YoloOptions { OnnxModel = Model5u_ObjectDetection }),
            YoloType.V5u_Obb_CPU => new Yolo(new YoloOptions { OnnxModel = Model5u_ObbDetection, Cuda = false }),
            YoloType.V5u_Obb_GPU => new Yolo(new YoloOptions { OnnxModel = Model5u_ObbDetection }),
            YoloType.V5u_Pos_CPU => new Yolo(new YoloOptions { OnnxModel = Model5u_PoseEstimation, Cuda = false }),
            YoloType.V5u_Pos_GPU => new Yolo(new YoloOptions { OnnxModel = Model5u_PoseEstimation }),
            YoloType.V5u_Seg_CPU => new Yolo(new YoloOptions { OnnxModel = Model5u_Segmentation, Cuda = false }),
            YoloType.V5u_Seg_GPU => new Yolo(new YoloOptions { OnnxModel = Model5u_Segmentation }),

            YoloType.V8_Cls_CPU => new Yolo(new YoloOptions { OnnxModel = Model8_Classification, Cuda = false }),
            YoloType.V8_Cls_GPU => new Yolo(new YoloOptions { OnnxModel = Model8_Classification }),
            YoloType.V8_Obj_CPU => new Yolo(new YoloOptions { OnnxModel = Model8_ObjectDetection, Cuda = false }),
            YoloType.V8_Obj_GPU => new Yolo(new YoloOptions { OnnxModel = Model8_ObjectDetection }),
            YoloType.V8_Obb_CPU => new Yolo(new YoloOptions { OnnxModel = Model8_ObbDetection, Cuda = false }),
            YoloType.V8_Obb_GPU => new Yolo(new YoloOptions { OnnxModel = Model8_ObbDetection }),
            YoloType.V8_Pos_CPU => new Yolo(new YoloOptions { OnnxModel = Model8_PoseEstimation, Cuda = false }),
            YoloType.V8_Pos_GPU => new Yolo(new YoloOptions { OnnxModel = Model8_PoseEstimation }),
            YoloType.V8_Seg_CPU => new Yolo(new YoloOptions { OnnxModel = Model8_Segmentation, Cuda = false }),
            YoloType.V8_Seg_GPU => new Yolo(new YoloOptions { OnnxModel = Model8_Segmentation }),
            
            YoloType.V9_Cls_CPU => new Yolo(new YoloOptions { OnnxModel = Model9_Classification, Cuda = false }),
            YoloType.V9_Cls_GPU => new Yolo(new YoloOptions { OnnxModel = Model9_Classification }),
            YoloType.V9_Obj_CPU => new Yolo(new YoloOptions { OnnxModel = Model9_ObjectDetection, Cuda = false }),
            YoloType.V9_Obj_GPU => new Yolo(new YoloOptions { OnnxModel = Model9_ObjectDetection }),
            YoloType.V9_Obb_CPU => new Yolo(new YoloOptions { OnnxModel = Model9_ObbDetection, Cuda = false }),
            YoloType.V9_Obb_GPU => new Yolo(new YoloOptions { OnnxModel = Model9_ObbDetection }),
            YoloType.V9_Pos_CPU => new Yolo(new YoloOptions { OnnxModel = Model9_PoseEstimation, Cuda = false }),
            YoloType.V9_Pos_GPU => new Yolo(new YoloOptions { OnnxModel = Model9_PoseEstimation }),
            YoloType.V9_Seg_CPU => new Yolo(new YoloOptions { OnnxModel = Model9_Segmentation, Cuda = false }),
            YoloType.V9_Seg_GPU => new Yolo(new YoloOptions { OnnxModel = Model9_Segmentation }),

            YoloType.V10_Cls_CPU => new Yolo(new YoloOptions { OnnxModel = Model10_Classification, Cuda = false }),
            YoloType.V10_Cls_GPU => new Yolo(new YoloOptions { OnnxModel = Model10_Classification }),
            YoloType.V10_Obj_CPU => new Yolo(new YoloOptions { OnnxModel = Model10_ObjectDetection, Cuda = false }),
            YoloType.V10_Obj_GPU => new Yolo(new YoloOptions { OnnxModel = Model10_ObjectDetection }),
            YoloType.V10_Obb_CPU => new Yolo(new YoloOptions { OnnxModel = Model10_ObbDetection, Cuda = false }),
            YoloType.V10_Obb_GPU => new Yolo(new YoloOptions { OnnxModel = Model10_ObbDetection }),
            YoloType.V10_Pos_CPU => new Yolo(new YoloOptions { OnnxModel = Model10_PoseEstimation, Cuda = false }),
            YoloType.V10_Pos_GPU => new Yolo(new YoloOptions { OnnxModel = Model10_PoseEstimation }),
            YoloType.V10_Seg_CPU => new Yolo(new YoloOptions { OnnxModel = Model10_Segmentation, Cuda = false }),
            YoloType.V10_Seg_GPU => new Yolo(new YoloOptions { OnnxModel = Model10_Segmentation }),
            
            YoloType.V11_Cls_CPU => new Yolo(new YoloOptions { OnnxModel = Model11_Classification, Cuda = false }),
            YoloType.V11_Cls_GPU => new Yolo(new YoloOptions { OnnxModel = Model11_Classification }),
            YoloType.V11_Obj_CPU => new Yolo(new YoloOptions { OnnxModel = Model11_ObjectDetection, Cuda = false }),
            YoloType.V11_Obj_GPU => new Yolo(new YoloOptions { OnnxModel = Model11_ObjectDetection }),
            YoloType.V11_Obb_CPU => new Yolo(new YoloOptions { OnnxModel = Model11_ObbDetection, Cuda = false }),
            YoloType.V11_Obb_GPU => new Yolo(new YoloOptions { OnnxModel = Model11_ObbDetection }),
            YoloType.V11_Pos_CPU => new Yolo(new YoloOptions { OnnxModel = Model11_PoseEstimation, Cuda = false }),
            YoloType.V11_Pos_GPU => new Yolo(new YoloOptions { OnnxModel = Model11_PoseEstimation }),
            YoloType.V11_Seg_CPU => new Yolo(new YoloOptions { OnnxModel = Model11_Segmentation, Cuda = false }),
            YoloType.V11_Seg_GPU => new Yolo(new YoloOptions { OnnxModel = Model11_Segmentation }),

            YoloType.V12_Cls_CPU => new Yolo(new YoloOptions { OnnxModel = Model11_Classification, Cuda = false }),
            YoloType.V12_Cls_GPU => new Yolo(new YoloOptions { OnnxModel = Model11_Classification }),
            YoloType.V12_Obj_CPU => new Yolo(new YoloOptions { OnnxModel = Model11_ObjectDetection, Cuda = false }),
            YoloType.V12_Obj_GPU => new Yolo(new YoloOptions { OnnxModel = Model11_ObjectDetection }),
            YoloType.V12_Obb_CPU => new Yolo(new YoloOptions { OnnxModel = Model11_ObbDetection, Cuda = false }),
            YoloType.V12_Obb_GPU => new Yolo(new YoloOptions { OnnxModel = Model11_ObbDetection }),
            YoloType.V12_Pos_CPU => new Yolo(new YoloOptions { OnnxModel = Model11_PoseEstimation, Cuda = false }),
            YoloType.V12_Pos_GPU => new Yolo(new YoloOptions { OnnxModel = Model11_PoseEstimation }),
            YoloType.V12_Seg_CPU => new Yolo(new YoloOptions { OnnxModel = Model11_Segmentation, Cuda = false }),
            YoloType.V12_Seg_GPU => new Yolo(new YoloOptions { OnnxModel = Model11_Segmentation }),

            _ => throw new ArgumentOutOfRangeException()
        };
    }
}
