#!/usr/bin/env python3
import os
import cv2
import logging
import argparse
import subprocess
import scenedetect
import moviepy.editor as mpy
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
import openai
import numpy as np
from multiprocessing import Pool

# print(os.environ['PATH'])  # Remove or comment out this line

# ---------- VIDEO SPLITTING ----------

def is_dark_frame(frame, threshold=30):
    """Check if frame is too dark based on average pixel intensity."""
    return cv2.mean(frame)[0] < threshold

def analyze_scene_content(args):
    """Analyze scene content with minimal processing."""
    video_path, start_time, duration = args
    try:
        video = cv2.VideoCapture(video_path)
        video.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

        # Only analyze 5 frames per clip instead of continuous analysis
        frames_to_check = 5
        frame_scores = []

        for _ in range(frames_to_check):
            ret, frame = video.read()
            if not ret:
                break

            # Simple brightness and edge detection (much faster)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            edges = cv2.Canny(gray, 100, 200)
            edge_score = np.mean(edges)

            score = brightness * 0.4 + edge_score * 0.6
            frame_scores.append(score)

            # Skip frames to speed up analysis
            video.set(cv2.CAP_PROP_POS_FRAMES, video.get(cv2.CAP_PROP_POS_FRAMES) + 30)

        video.release()
        return (start_time, np.mean(frame_scores) if frame_scores else 0)
    except Exception as e:
        logging.error(f"Error in analyze_scene_content: {e}")
        return (start_time, 0)

def split_video_fixed_duration(video_path, clip_duration):
    """Split video into 3 clips with high quality."""
    try:
        print("Opening video file...")
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise Exception("Could not open video file")

        total_duration = video.get(cv2.CAP_PROP_FRAME_COUNT) / video.get(cv2.CAP_PROP_FPS)
        print(f"Video duration: {total_duration:.2f} seconds")

        # Sample fewer points (every 60 seconds instead of 30)
        start_offset = 60
        end_offset = 60
        sample_points = [(video_path, time, clip_duration)
                        for time in range(int(start_offset),
                                        int(total_duration - end_offset - clip_duration),
                                        60)]

        print("Analyzing scenes (this may take a few minutes)...")
        # Use smaller process pool to avoid memory issues
        with Pool(processes=4) as pool:
            results = pool.map(analyze_scene_content, sample_points[:30])  # Limit analysis to first 30 points

        # Sort by score and take top 3
        results.sort(key=lambda x: x[1], reverse=True)
        best_points = results[:3]
        best_points.sort(key=lambda x: x[0])

        clip_paths = []
        for i, (start_time, score) in enumerate(best_points):
            output_path = f"clip_{i}.mp4"
            print(f"\nExtracting clip {i+1}/3 from {start_time:.2f}s")

            # High quality FFmpeg command
            cmd = (
                f'ffmpeg -y -ss {start_time} -i "{video_path}" -t {clip_duration} '
                f'-c:v libx264 -preset medium -crf 18 -profile:v high '
                f'-c:a aac -b:a 192k "{output_path}"'
            )

            print(f"Running FFmpeg command...")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Created {output_path}")
                clip_paths.append(output_path)
            else:
                print(f"Error: {result.stderr}")

        video.release()
        return clip_paths

    except Exception as e:
        print(f"Error: {str(e)}")
        logging.error(f"Error: {str(e)}", exc_info=True)
        raise

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

def format_timecode(seconds):
    """Convert seconds to SRT timecode format with millisecond precision."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    msecs = int((secs - int(secs)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(secs):02d},{msecs:03d}"

def overlay_subtitles(video_file, subtitles):
    """Overlays subtitles using MoviePy with precise timing."""
    try:
        video = mpy.VideoFileClip(video_file)
        subtitle_clips = []

        for sub in subtitles:
            # Convert time to seconds if it's not already
            if isinstance(sub['start'], str):
                start_time = sum(float(x) * 60 ** i for i, x in enumerate(reversed(sub['start'].split(':'))))
                end_time = sum(float(x) * 60 ** i for i, x in enumerate(reversed(sub['end'].split(':'))))
            else:
                start_time = sub['start']
                end_time = sub['end']

            text_clip = mpy.TextClip(
                sub['text'],
                font='Arial',
                fontsize=24,
                color='white',
                bg_color='black',
                size=(video.w * 0.8, None),  # 80% of video width
                method='caption'
            )
            text_clip = text_clip.set_position(('center', 'bottom'))
            text_clip = text_clip.set_start(start_time).set_end(end_time)
            subtitle_clips.append(text_clip)

        final = mpy.CompositeVideoClip([video] + subtitle_clips)
        output_path = f"subtitled_{os.path.basename(video_file)}"
        final.write_videofile(output_path, codec="libx264", audio_codec="aac")
        return output_path
    except Exception as e:
        logging.error(f"Error in overlay_subtitles: {e}")
        raise

def format_ass_time(seconds):
    """Convert seconds to ASS time format (h:mm:ss.cc)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    centisecs = int((secs - int(secs)) * 100)
    return f"{hours}:{minutes:02d}:{int(secs):02d}.{centisecs:02d}"

# ---------- TITLE GENERATION ----------

def generate_title_with_gpt4(transcript, api_key):
    """Generates a title using GPT-4 based on the transcript."""
    openai.api_key = api_key
    response = openai.Completion.create(
        engine="text-davinci-003",  # Adjust if using a GPT-4 engine
        prompt=f"Generate a short, engaging title for a video with the following transcript:\n\n{transcript}",
        max_tokens=50,
        temperature=0.7,
    )
    title = response.choices[0].text.strip()
    return title

def extract_keywords(text):
    """A dummy keyword extraction function.
       Replace with your own keyword extraction logic."""
    words = text.split()
    # Return the first 3 words as keywords for demonstration
    return words[:3]

def generate_title_heuristic(transcript):
    """Generates a title using heuristic rules (e.g., keywords)."""
    keywords = extract_keywords(transcript)
    title = " | ".join(keywords)
    return title

# ---------- YOUTUBE UPLOADING ----------

def authenticate_youtube_api(api_key):
    """Authenticates with the YouTube API using an API key."""
    youtube = build('youtube', 'v3', developerKey=api_key)
    return youtube

def upload_to_youtube(youtube, video_file, title, description, category='22', privacy_status='public'):
    """Uploads the video to YouTube."""
    body = {
        'snippet': {
            'title': title,
            'description': description,
            'categoryId': category
        },
        'status': {
            'privacyStatus': privacy_status
        }
    }
    media = MediaFileUpload(video_file, mimetype='video/mp4', resumable=True)
    request = youtube.videos().insert(
        part='snippet,status',
        body=body,
        media_body=media
    )
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"Uploaded {int(status.progress() * 100)}%")
    print("Upload complete!")
    return response

# ---------- PLACEHOLDER FOR VIDEO TRANSCRIPTION ----------

def transcribe_video(video_file):
    """
    Dummy transcription function.
    Replace this with a call to an actual transcription service (e.g., OpenAI Whisper).
    """
    # For demonstration, we return a fixed string.
    return "This is a sample transcript of the video content."

# ---------- MAIN PROCESSING FUNCTION ----------

def check_ffmpeg():
    """Check if FFmpeg is available in the system."""
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        print("Error: FFmpeg is not installed or not in PATH")
        print("Please install FFmpeg from https://ffmpeg.org/download.html")
        print("And add it to your system PATH")
        return False

def main(video_path, clip_duration, subtitle_file, api_key):
    """Orchestrates the video processing pipeline."""
    try:
        if not check_ffmpeg():
            return

        print("Splitting video into clips...")
        clip_paths = split_video_fixed_duration(video_path, clip_duration)

        if subtitle_file:
            print("Loading subtitles...")
            all_subtitles = extract_subtitles_from_srt(subtitle_file)

        processed_clips = []
        for i, clip_path in enumerate(clip_paths):
            print(f"\nProcessing clip {i+1}/{len(clip_paths)}...")

            if subtitle_file:
                clip_start = i * clip_duration
                clip_subtitles = extract_subtitles_for_clip(all_subtitles, clip_start, clip_duration)

                if clip_subtitles:
                    print(f"Adding {len(clip_subtitles)} subtitles to clip...")
                    output_path = overlay_subtitles(clip_path, clip_subtitles)
                    processed_clips.append(output_path)
                else:
                    print("No subtitles found for this clip segment")
                    processed_clips.append(clip_path)
            else:
                processed_clips.append(clip_path)

        # Upload clips to YouTube
        youtube = authenticate_youtube_api(api_key)
        for i, clip_path in enumerate(processed_clips):
            print(f"\nUploading clip {i+1}/{len(processed_clips)}...")
            transcript = transcribe_video(clip_path)
            title = generate_title_with_gpt4(transcript, api_key)
            description = f"Auto-generated clip from video\nTranscript:\n{transcript}"
            upload_to_youtube(youtube, clip_path, title, description)

        # Clean up temporary files
        for path in clip_paths + [p for p in processed_clips if p not in clip_paths]:
            try:
                os.remove(path)
                print(f"Cleaned up {path}")
            except:
                pass

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        print("An error occurred. Check the log for details.")

# ---------- ARGUMENT PARSING ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Video Processing and YouTube Upload")
    parser.add_argument("--video_path", required=True, help="Path to the input video file")
    parser.add_argument("--clip_duration", type=int, default=30,
                       help="Duration (in seconds) for fixed clip splitting. Set to 0 to use scene detection.")
    parser.add_argument("--subtitle_file", help="Path to the subtitle SRT file", default=None)
    parser.add_argument("--api_key", help="API key for GPT-4 title generation and YouTube upload", required=True)

    args = parser.parse_args()
    main(args.video_path, args.clip_duration, args.subtitle_file, args.api_key)
