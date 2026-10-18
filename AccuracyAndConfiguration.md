# 💡 Accuracy Depends on Configuration

**Detection accuracy is not automatic — it is the result of matching inference-time configuration to how the model was trained.**  
Even a correctly trained model can produce poor results if preprocessing, resize behavior, or thresholds do not align with training assumptions.

There is no one-size-fits-all configuration. Optimal settings depend on your dataset, training pipeline, and application requirements.

### 🔑 Key Factors That Affect Accuracy
1. **Image Preprocessing & Resize Mode**
    * Controlled via `ImageResize`.
    * Resize behavior must **match the preprocessing used during training**. A mismatch here is one of the most common causes of degraded detection accuracy.
        | **Proportional dataset** | **Stretched dataset** |
        |:------------:|:---------:|
        |<img width="160" height="107" alt="proportional" src="https://github.com/user-attachments/assets/e95a2c5c-8032-4dee-a05a-a73b062a4965" />|<img width="160" height="160" alt="stretched" src="https://github.com/user-attachments/assets/90fa31cf-89dd-41e4-871c-76ae3215171f" />|
        |Use `ImageResize.Proportional` (default) when the training dataset preserved aspect ratio (letterboxing or padding).|Use `ImageResize.Stretched` if images were resized directly to the model input size (e.g. 640×640) without preserving aspect ratio.|
    
    > ℹ️ Most Ultralytics YOLO models are trained using proportional resizing with padding (letterboxing).
    > `ImageResize.Proportional` is the default in YoloDotNet and is the correct choice in the majority of cases.

    > **Important:**\
    > Selecting the wrong resize mode can reduce detection accuracy. Change this only if you know your training pipeline resized images without preserving aspect ratio.

2. **Sampling Options**
    * Controlled via `SamplingOptions`.
    * Define how pixel data is resampled when resizing (e.g., **`Cubic`**, **`NearestNeighbor`**, **`Bilinear`**). This choice has a direct impact on the accuracy of your detections, as different resampling methods can slightly alter object shapes and edges.
    * YoloDotNet default:
        ```csharp
        SamplingOptions = new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.None);
        ```
        Bilinear resampling is what YOLO models are trained and validated with, so this default gives the most
        accurate detections. 32-bit BGRA/RGBA images (decoded files, video frames, bitmaps created in code) are resized
        with YoloDotNet's fast AVX2 path, which is about as fast as nearest-neighbour. Other sampling methods are drawn
        by SkiaSharp exactly as configured, and can subtly change detection results.

    * **Transparency:** like the training pipeline, YoloDotNet ignores the alpha channel of 32-bit images and uses
      each pixel's stored colour. Opaque images are unaffected. SkiaSharp decodes images with transparency
      premultiplied by default, which darkens transparent areas (fully transparent pixels become black). To give the
      model exactly what it was trained on, decode such images unpremultiplied:
        ```csharp
        using var codec = SKCodec.Create(path);
        using var bitmap = SKBitmap.Decode(codec, codec.Info.WithAlphaType(SKAlphaType.Unpremul));
        ```

        💡 **Tip:** Check the [ResizeImage Benchmarks](./test/YoloDotNet.Benchmarks/ImageBenchmarks/ImageResizeBenchmarks.cs) for examples of different `SamplingOptions` and to help you choose the best settings for your needs.

3. **Confidence & IoU Thresholds**
    
    Detection results are filtered using thresholds you configure at inference time:
    - **Confidence**\
            Minimum probability required for a detection to be considered valid.
    - **IoU (Intersection-over-Union)**\
            Controls how overlapping detections are merged or suppressed.

    General guidance:
    * Too low → more false positives.
    * Too high → more missed detections.

    Fine-tune these values based on your dataset, object density, and tolerance for false positives.

> 💡 **Recommended approach**:
> Start with the defaults, verify results on a representative validation set, then adjust `ImageResize`, `SamplingOptions`, and `Confidence/IoU` thresholds incrementally.

# Visualization & Styling (Optional)
Visualization does not affect inference accuracy, but YoloDotNet provides full control over how results are rendered.

- **Custom Colors** – use built-in class colors or define your own
- **Font Style & Size** – fully configurable label rendering
- **Custom Fonts** – load your own font files for overlays

For advanced customization, see the extension methods in the main YoloDotNet repository — they provide a solid, production-ready foundation.

[Practical drawing and rendering examples are available in the demo project source code](/Demo/).