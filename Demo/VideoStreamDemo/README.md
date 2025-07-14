# Demo - Video Streams

**Note:** For detailed configuration options and usage guidance, please refer to the inline comments in the demo’s source files.

The Stream Video demo demonstrates object detection and tracking on videos or livestreams using the YoloDotNet library.
This demo loads a video source (file, livestream, or webcam), runs object detection and optional tracking on each frame,
draws the results (bounding boxes, labels, confidence scores, and tracked tails), and optionally saves the output video.

It showcases:
- Model initialization with configurable hardware and preprocessing options
- Real-time object detection using YoloDotNet on video streams
- Filtering detections by class labels
- Multi-object tracking across frames using the SORT tracker
- Rendering detection and tracking results directly on video frames
- Saving processed video output and optionally splitting into chunks
- Progress reporting and end-of-stream handling with customizable callbacks

Example video sources:
- Local file        (e.g., C:\videos\test.mp4)
- Livestream        (e.g., rtmp://your.server/stream)
- Webcam (Windows)  (e.g., device:Logitech BRIO)
- Webcam (Linux)    (e.g., "device:/dev/video0)

Note:
- CUDA acceleration is required.
- FFmpeg and FFprobe must be added to your system PATH variable -> [Download and install](https://ffmpeg.org/download.html)
- The demo creates an output folder on the desktop to store processed results.

# Example Use Cases
This demo can be used for virtually any application involving real-time video object detection and tracking — from personal experiments to production-grade systems. Below are some practical examples of how you might use it:

### 🛡️ Surveillance and Security
- Detect and track moving objects
- Monitor movement across defined zones (e.g., entryways, parking lots, or restricted areas)
- Record annotated video for auditing or post-event analysis

### 🏭 Industrial Automation
- Detect human presence in hazardous machine zones
- Monitor conveyor belts or assembly lines for object presence
- Count or verify packages in transit

### 🚦 Smart Cities and Traffic Monitoring
- Track pedestrian flow at crosswalks or public areas
- Analyze vehicle movements at intersections or highways
- Detect stationary objects in no-parking zones

### 🎥 Live Streaming with AI Overlays
- Broadcast annotated livestreams with real-time object detection (e.g., sports, wildlife cams)
- Add AI-powered overlays to webcam streams for content creators

### 🧪 Research and Prototyping
- Benchmark model performance on video datasets
- Collect inference results for downstream analysis or custom logic
- Easily test model variations with different confidence/IoU thresholds