# <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/994287a9-556c-495f-8acf-1acae8d64ac0" height=24> Docker Demo
**Object detection as a Web API in a Docker container**

A minimal ASP.NET Core Web API that runs **YoloDotNet** object detection on the CPU inside a **Docker container**.
Send it an image and it returns the detected objects as JSON, or the image with the results drawn on it.

This demo is a **getting-started reference**, not a production-ready service: it keeps the setup small and explicit,
and runs on the CPU so it works in any container without GPU setup.

## ▶️ Run the demo

You need [Docker](https://www.docker.com/) and the exported models; see the [Demo README](../README.md#-quick-start).
Step 1 is run from this demo's folder (`Demo/DockerDemo`); the Docker steps are run from the **repository root**.

### 1. Build the project in Release mode

```bash
dotnet build -c Release
```

> [!IMPORTANT]
> The project **must** be built in **Release** mode: the Docker image copies the Release build output (`bin/Release/net8.0`).

### 2. The model

The Dockerfile copies `yolo26s.onnx` (object detection) from the repository's `test/assets/Models` folder
into the image as `models/model.onnx`, which is what the API loads at startup.

> [!IMPORTANT]
> Make sure `test/assets/Models/yolo26s.onnx` exists **before** building the image — the models are not committed
> to the repo. If they are missing, run `python tools/download_models.py` from the repository root to download
> and export them into that folder.

### 3. Build the Docker image

Run from the **repository root** (the build context must include both the demo's build output and the models folder):

```bash
docker build -f Demo/DockerDemo/Dockerfile -t yolodotnet:demo .
```

### 4. Run the container

```bash
docker run -p 8080:8080 yolodotnet:demo
```

The API is available at `http://localhost:8080`.

## 🧩 How it works

[`Program.cs`](./Program.cs) checks that the model exists, loads it once at startup and shares it between requests
(inference on one `Yolo` instance is thread-safe):

```csharp
builder.Services.AddSingleton(_ => new Yolo(new YoloOptions
{
    ExecutionProvider = new CpuExecutionProvider(_modelPath)
}));
```

It exposes one endpoint, `POST /detect`, which:

**1. Reads the uploaded image** from the `image` form field.

**2. Runs inference** with `RunObjectDetection`, which returns a list of `ObjectDetection` results.

```csharp
var detections = yolo.RunObjectDetection(image);
```

**3. Returns the results** as JSON, or, with `?download=true`, the image with the results drawn on it as a JPEG.

The [`Dockerfile`](./Dockerfile) starts from the .NET 8 ASP.NET runtime image, installs the native libraries SkiaSharp needs on Linux,
copies the Release build output along with `yolo26s.onnx` from `test/assets/Models` (as `models/model.onnx`), and listens on port 8080.

## 🌐 API

### Request

```
POST /detect
```

- **Content-Type:** `multipart/form-data`
- **Form field:** `image`, the input image file (jpg, png, …)

### JSON response (detection results)

```bash
curl -F "image=@street.jpg" http://localhost:8080/detect
```

Returns each detected object with its label, confidence and bounding box (in pixels):

```json
[
  {
    "label": "person",
    "confidence": 0.92,
    "bbox": {
      "x": 120,
      "y": 45,
      "width": 210,
      "height": 430
    }
  }
]
```

### Image response (detections drawn)

```bash
curl -F "image=@street.jpg" "http://localhost:8080/detect?download=true" -o result.jpg
```

Returns the input image with the detected objects drawn on it (`Content-Type: image/jpeg`).

### Errors

- `400 Bad Request` if the request isn't `multipart/form-data`, the `image` field is missing or empty, or the image can't be decoded.
- `500` with the error message if detection fails.

## ⚙️ Options

### Execution provider

The API uses `CpuExecutionProvider(model)`, which runs on any machine and in any container.
For other providers, see [Execution Providers](../../README.md#️-execution-providers) in the main README.

### Detection (`RunObjectDetection`)

The API uses the defaults:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `confidence` | `0.2` | Minimum confidence for an object to be returned. |
| `iou` | `0.7` | Overlap threshold for removing duplicate boxes (non-maximum suppression). |
| `roi` | `null` | Optional region of interest (`SKRectI`); only objects inside it are detected. |

📖 For guidance on choosing these values, see the [Accuracy & Configuration Guide](../../AccuracyAndConfiguration.md).

### Drawing

With `?download=true`, the API draws with the default `DetectionDrawingOptions`. For all drawing options and their defaults,
see [Drawing](../ObjectDetectionDemo/README.md#drawing-detectiondrawingoptions) in the ObjectDetectionDemo README.

## 📚 See also

- [Demo README](../README.md): setup and the steps all demos share
- [ObjectDetectionDemo](../ObjectDetectionDemo/README.md): object detection on an image, and all drawing options
- [BatchDemo](../BatchDemo): processing many images in parallel with one `Yolo` instance
