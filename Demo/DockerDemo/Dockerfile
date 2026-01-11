# .NET 8 ASP.NET runtime (no SDK)
FROM mcr.microsoft.com/dotnet/aspnet:8.0

# SkiaSharp native runtime dependencies (CPU, headless)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        libnuma1 \
        libfontconfig1 \
        libfreetype6 \
        libx11-6 \
        libxrender1 \
        libxcb1 \
        libxrandr2 \
    && rm -rf /var/lib/apt/lists/*

# libgomp1      – enables Skia to safely use multiple CPU cores for faster image processing
# libnuma1      – required by Skia’s native code to allocate and manage memory correctly
# libfontconfig – allows Skia to locate and select installed system fonts
# libfreetype   – allows Skia to draw text by rasterizing font files (TTF/OTF)
# libx11        – required system library that Skia expects to be present on Linux
# libxrender    – provides rendering-related symbols used internally by Skia
# libxcb        – low-level X11 dependency required by other X11 libraries
# libxrandr     – provides screen and display information APIs required by Skia

WORKDIR /app

# Copy published output
COPY bin/Release/net8.0/ .

# Runtime configuration
ENV MODEL_PATH=/app/models/model.onnx
ENV ASPNETCORE_URLS=http://+:8080

EXPOSE 8080

# Start application
ENTRYPOINT ["dotnet", "DockerDemo.dll"]