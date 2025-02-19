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



fix this !

# ---------- SUBTITLE EXTRACTION & OVERLAY ----------

def extract_subtitles_from_srt(srt_file):
    """Extracts subtitles with precise timing from an SRT file."""
    subtitles = []
    current_sub = {}
    try:
        with open(srt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if '-->' in line:
                try:
                    times = line.split(' --> ')
                    current_sub['start'] = times[0].strip()
                    current_sub['end'] = times[1].strip()
                except Exception as e:
                    logging.error(f"Error parsing timecodes: {e}")
                    continue
            elif line and not line.isdigit():
                if 'text' not in current_sub:
                    current_sub['text'] = line
                else:
                    current_sub['text'] += '\n' + line
            elif not line and 'text' in current_sub:
                subtitles.append(current_sub.copy())
                current_sub = {}

    except Exception as e:
        logging.error(f"Error reading SRT file: {e}")
        return []

    return subtitles

def extract_subtitles_for_clip(subtitles, clip_start, clip_duration):
    """Extract subtitles with precise timing alignment."""
    clip_subtitles = []
    clip_end = clip_start + clip_duration

    for sub in subtitles:
        try:
            # Parse original timecodes more precisely
            start_parts = sub['start'].split(':')
            end_parts = sub['end'].split(':')
            
            # Calculate precise timestamps in seconds
            sub_start = (int(start_parts[0]) * 3600 + 
                        int(start_parts[1]) * 60 + 
                        float(start_parts[2].replace(',', '.')))
            
            sub_end = (int(end_parts[0]) * 3600 + 
                      int(end_parts[1]) * 60 + 
                      float(end_parts[2].replace(',', '.')))

            # Check if subtitle overlaps with clip
            if sub_end > clip_start and sub_start < clip_end:
                adjusted_sub = sub.copy()
                
                # Adjust timing relative to clip start
                adjusted_sub['start'] = max(0, sub_start - clip_start)
                adjusted_sub['end'] = min(clip_duration, sub_end - clip_start)
                
                if adjusted_sub['end'] > adjusted_sub['start']:
                    clip_subtitles.append(adjusted_sub)

        except Exception as e:
            logging.error(f"Error processing subtitle: {e}")
            continue

    return clip_subtitles

def overlay_subtitles(video_file, subtitles):
    """Overlay subtitles using FFmpeg directly."""
    try:
        output_path = f"subtitled_{os.path.basename(video_file)}"
        
        # Create temporary SRT file
        temp_srt = "temp_subtitles.srt"
        with open(temp_srt, 'w', encoding='utf-8') as f:
            for i, sub in enumerate(subtitles, 1):
                # Format timecodes
                start = format_timecode(float(sub['start']))
                end = format_timecode(float(sub['end']))
                # Write SRT entry
                f.write(f"{i}\n{start} --> {end}\n{sub['text']}\n\n")

        # FFmpeg command with subtitle burning
        cmd = (
            f'ffmpeg -y -i "{video_file}" '
            f'-vf "subtitles={temp_srt}:force_style=\'FontName=Arial,FontSize=24,PrimaryColour=&H00000000,OutlineColour=&H00000000,BorderStyle=1,Outline=0,Shadow=0,BackColour=&H00000000\'" '
            f'-c:v libx264 -preset medium -crf 18 '
            f'-c:a copy "{output_path}"'
        )

        print(f"Adding subtitles to {video_file}...")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # Clean up temporary file
        if os.path.exists(temp_srt):
            os.remove(temp_srt)
        
        if result.returncode == 0:
            print(f"Successfully created {output_path}")
            return output_path
        else:
            print(f"Error burning subtitles: {result.stderr}")
            return video_file
            
    except Exception as e:
        logging.error(f"Error in overlay_subtitles: {e}")
        if os.path.exists(temp_srt):
            os.remove(temp_srt)
        return video_file

def format_timecode(seconds):
    """Convert seconds to SRT timecode format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    msecs = int((seconds * 1000) % 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{msecs:03d}"

def format_ass_time(seconds):
    """Convert seconds to ASS time format (h:mm:ss.cc)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    centisecs = int((secs - int(secs)) * 100)
    return f"{hours}:{minutes:02d}:{int(secs):02d}.{centisecs:02d}"
