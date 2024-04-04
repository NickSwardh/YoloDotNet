namespace YoloDotNet.Configuration
{
    public static class ImageConfig
    {
        public const float DEFAULT_FONT_SIZE = 16f;

        public const float FONT_SIZE_8 = 8f;

        public const float LINE_SPACING = 1.5f;

        public const float SHADOW_OFFSET = 1;



        public static readonly Rgba32 SHADOW_COLOR = new(0, 0, 0, 60);

        public static readonly Rgba32 FOREGROUND_COLOR = new(255, 255, 255);



        public const int DEFAULT_OPACITY = 128;

        public const int POSE_ESTIMATION_MARKER_OPACITY = 192;

        public const float SEGMENTATION_MASK_OPACITY = .38f;



        public const float SEGMENTATION_PIXEL_THRESHOLD = .68f;


        // Image size for GPU memory allocation.
        public const int GPU_IMG_ALLOC_SIZE = 1080;
    }
}
