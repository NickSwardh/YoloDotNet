# YoloDotNet Docker Demo

A minimal ASP.NET Core Web API demonstrating how to run **YoloDotNet** inside a **Docker container** for CPU-based object detection using an ONNX YOLO model.

This project is intended as a **getting-started reference** and keeps the setup intentionally simple and explicit.

## Overview

This demo:

- Runs object detection using **YoloDotNet** with a CPU execution provider
- Uses a **YOLOv11 ONNX model**
- Exposes a single HTTP endpoint for image inference
- Is fully containerized using Docker

## Build and Run (Docker)

### 1. Build the project in Release mode

Build the project in **Release** mode so all required binaries are available:

```bash
dotnet build -c Release
```

> ℹ️ **Important:**  
> The YOLO model is included automatically as part of the project output.  
> The project **must** be built in **Release** mode, as the Docker image relies on the Release build output.

### 2. Build the Docker image

From the root of this demo project (where the Dockerfile is located):

```bash
docker build -t yolodotnet:demo .
```

### 3. Run the container

```bash
docker run -p 8080:8080 yolodotnet:demo
```

The API will be available at:

```bash
http://localhost:8080
```

## API Usage
### Endpoint

```bash
POST /detect
```

### Request

- **Content-Type:** multipart/form-data
- **Form field:**
  - `image` – input image file (jpg, png, etc.)

## Examples
### JSON response (detection results)

```bash
POST http://localhost:8080/detect
```

Returns detection results as JSON, including:

- Label
- Confidence
- Bounding box coordinates

**Example:**
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

### Image download with detections drawn

```bash
POST http://localhost:8080/detect?download=true
```

Response:
```
Content-Type: image/jpeg
```

Returns the input image with detected objects drawn directly on top.