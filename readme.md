# AutoCutMark Video Processing Tool

A Python tool that automatically processes videos by creating clips, adding subtitles, and uploading them to YouTube.

## Features

- Automatically skips dark intro scenes
- Creates 3 random clips from the video
- Adds subtitles from SRT files
- Generates engaging titles using GPT-4
- Uploads processed clips to YouTube

## Prerequisites

- Python 3.6+
- FFmpeg installed and in system PATH
- YouTube API credentials
- OpenAI API key for GPT-4

### Required Python Packages

```bash
pip install opencv-python
pip install moviepy
pip install google-api-python-client
pip install google-auth-oauthlib
pip install openai
pip install scenedetect
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/autocutmark.git
cd autocutmark
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install FFmpeg:
   - Download from [FFmpeg official website](https://ffmpeg.org/download.html)
   - Add to system PATH

## Usage

Basic usage with minimal parameters:

```bash
python video_processing.py --video_path "path/to/video.mp4" --api_key "your-api-key"
```

Full usage with all options:

```bash
python video_processing.py \
    --video_path "path/to/video.mp4" \
    --clip_duration 50 \
    --subtitle_file "path/to/subtitles.srt" \
    --api_key "your-api-key"
```

### Parameters

- `--video_path`: Path to the input video file (required)
- `--clip_duration`: Duration of each clip in seconds (default: 30)
- `--subtitle_file`: Path to the SRT subtitle file (optional)
- `--api_key`: YouTube API key for upload and GPT-4 for title generation (required)

## How It Works

1. **Video Processing**:
   - Detects and skips dark intro scenes
   - Creates 3 random clips of specified duration
   - Analyzes scenes for interesting content

2. **Subtitle Integration**:
   - Parses SRT subtitle files
   - Adjusts subtitle timing for clips
   - Overlays subtitles on video clips

3. **Title Generation**:
   - Uses GPT-4 to generate engaging titles
   - Extracts keywords from video content

4. **YouTube Upload**:
   - Authenticates with YouTube API
   - Uploads processed clips
   - Sets titles and descriptions

## Error Handling

- Logs errors to `video_processing.log`
- Provides user-friendly error messages
- Checks for FFmpeg installation

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- FFmpeg for video processing
- OpenAI for GPT-4 integration
- YouTube API for video uploads
