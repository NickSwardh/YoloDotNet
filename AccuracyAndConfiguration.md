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
    
    > **Important:**\
    > Selecting the wrong resize mode can reduce detection accuracy.

2. **Sampling Options**
    * Controlled via `SamplingOptions`.
    * Define how pixel data is resampled when resizing (e.g., **`Cubic`**, **`NearestNeighbor`**, **`Bilinear`**). This choice has a direct impact on the accuracy of your detections, as different resampling methods can slightly alter object shapes and edges.
    * YoloDotNet default:
        ```csharp
        SamplingOptions = new SKSamplingOptions(SKFilterMode.Nearest, SKMipmapMode.None);
        ```
        💡 **Tip:** Check the [ResizeImage Benchmarks](./test/YoloDotNet.Benchmarks/ImageExtensionTests/ResizeImageTests.cs) for examples of different `SamplingOptions` and to help you choose the best settings for your needs.
3. **Confidence & IoU Thresholds**
    
    Detection results are filtered using thresholds you configure at inference time:
    * Results are filtered based on thresholds you set during inference:
        * **Confidence**\
            Minimum probability required for a detection to be considered valid.
        * **IoU (Intersection-over-Union)**\
            Controls how overlapping detections are merged or suppressed.

    General guidance:
    * Too low → more false positives.
    * Too high → more missed detections.

    Fine-tune these values based on your dataset, object density, and tolerance for false positives.

💡 **Recommended approach**: Start with the defaults, then adjust `ImageResize`, `SamplingOptions`, and `Confidence/IoU` thresholds based on your dataset for optimal detection results.

# Visualization & Styling (Optional)
Want to give your detections a personal touch? Go ahead! If you're drawing bounding boxes on-screen, there’s full flexibility to style them just the way you like:

- **Custom Colors** – use built-in class colors or define your own
- **Font Style & Size** – fully configurable label rendering
- **Custom Fonts** – load your own font files for overlays

For advanced customization, see the extension methods in the main YoloDotNet repository — they provide a solid, production-ready foundation.

[Practical drawing and rendering examples are available in the demo project source code](/Demo/).