namespace YoloDotNet.Configuration
{
    internal static class ImageConfig
    {
        public const float FONT_SIZE = 18;
        public const float KEYPOINT_SIZE = 5;
        public const float SHADOW_OFFSET = 1;
        public const float BORDER_THICKNESS = 2;

        public const float CLASSIFICATION_TRANSPARENT_BOX_X = 50;
        public const float CLASSIFICATION_TRANSPARENT_BOX_Y = 50;
        public const int CLASSIFICATION_BOX_ALPHA = 60;

        public const int DEFAULT_OPACITY = 128;

        public const int SEGMENTATION_MASK_OPACITY = 80;
        public const float SEGMENTATION_PIXEL_THRESHOLD = .68f;

        // Image size used for GPU memory allocation.
        public const int GPU_IMG_ALLOC_SIZE = 1080;

        public const int SCALING_DENOMINATOR = 1280;

        public static readonly SKColor ClassificationBackground
            = new(0, 0, 0, CLASSIFICATION_BOX_ALPHA);


        public static readonly SKColor TailPaintColorStart = SKColors.HotPink;
        public static readonly SKColor TailPaintColorEnd = TailPaintColorStart.WithAlpha(0);

        public static readonly SKPaint ResizePaint = new()
        {
            IsAntialias = false,
            IsDither = true
        };

        public static readonly SKPaint PaintFill = new()
        {
            Style = SKPaintStyle.Fill,
            IsAntialias = true
        };

        public static readonly SKPaint FontColorPaint = new()
        {
            Style = SKPaintStyle.Fill,
            IsAntialias = true,
            Color = SKColors.White
        };

        public static readonly SKPaint ClassificationBackgroundPaint = new ()
        {
            Style = SKPaintStyle.Fill,
            IsAntialias = true,
            Color = ClassificationBackground
        };

        public static readonly SKPaint TextShadowPaint = new ()
        {
            Color = new SKColor(0, 0, 0, DEFAULT_OPACITY),
            IsAntialias = true
        };
    }
}
