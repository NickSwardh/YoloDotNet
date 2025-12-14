namespace YoloDotNet.Core
{
    public ref struct InferenceResult
    {
        public ReadOnlySpan<float> OrtSpan0 { get; set; }
        public ReadOnlySpan<float> OrtSpan1 { get; set; }
        public SKSizeI ImageOriginalSize { get; set; }

        public InferenceResult()
        {
        }

        public InferenceResult(ReadOnlySpan<float> ortSpan0, ReadOnlySpan<float> ortSpan1, SKSizeI imageSize = default!)
        {
            OrtSpan0 = ortSpan0;
            OrtSpan1 = ortSpan1;
            ImageOriginalSize = imageSize;
        }
    }
}
