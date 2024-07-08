namespace YoloDotNet.Configuration
{
    public static class ImageConfig
    {
        public const float FONT_SIZE = 18;
        public const float FONT_SIZE_8 = 8;
        public const float SHADOW_OFFSET = 1;
        public const float BORDER_THICKNESS = 2;

        public const float CLASSIFICATION_TRANSPARENT_BOX_X = 50;
        public const float CLASSIFICATION_TRANSPARENT_BOX_Y = 50;
        public const int CLASSIFICATION_BOX_ALPHA = 60;

        public const int DEFAULT_OPACITY = 128;
        public const int POSE_ESTIMATION_MARKER_OPACITY = 192;

        public const float SEGMENTATION_MASK_OPACITY = .38f;
        public const float SEGMENTATION_PIXEL_THRESHOLD = .68f;

        // Image size used for GPU memory allocation.
        public const int GPU_IMG_ALLOC_SIZE = 1080;
    }
}
