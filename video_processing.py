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
    """Analyze scene content for interesting/funny moments."""
    video_path, start_time, duration = args
    try:
        video = cv2.VideoCapture(video_path)
        video.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

        frame_scores = []
        fps = video.get(cv2.CAP_PROP_FPS)
        sample_rate = 10  # Sample every 10 frames for speed
        frames_to_check = int(duration * fps / sample_rate)
        prev_frame = None

        for _ in range(frames_to_check):
            video.set(cv2.CAP_PROP_POS_FRAMES, video.get(cv2.CAP_PROP_POS_FRAMES) + sample_rate)
            ret, frame = video.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_frame is not None:
                # Motion detection (for action/funny physical scenes)
                motion_score = cv2.norm(cv2.absdiff(gray, prev_frame))

                # Face detection (for dialogue scenes)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                face_score = len(faces) * 1000  # Weight face detection heavily

                # Scene complexity (for interesting visual moments)
                edges = cv2.Canny(gray, 100, 200)
                edge_score = np.mean(edges)

                total_score = motion_score * 0.4 + face_score * 0.4 + edge_score * 0.2
                frame_scores.append(total_score)

            prev_frame = gray

        video.release()
        return (start_time, np.mean(frame_scores) if frame_scores else 0)
    except Exception as e:
        logging.error(f"Error in analyze_scene_content: {e}")
        return (start_time, 0)

def split_video_fixed_duration(video_path, clip_duration):
    """Split video into 3 interesting clips."""
    try:
        print("Opening video file...")
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise Exception("Could not open video file")

        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = total_frames / fps

        print(f"Video duration: {total_duration:.2f} seconds")

        # Skip intro
        start_offset = 60  # Skip first minute
        end_offset = 60    # Skip last minute
        usable_duration = total_duration - start_offset - end_offset

        # Sample points every 30 seconds
        sample_points = [(video_path, time, clip_duration) for time in range(int(start_offset), int(total_duration - end_offset - clip_duration), 30)]

        # Analyze scenes in parallel
        with Pool() as pool:
            results = pool.map(analyze_scene_content, sample_points)

        # Sort by score and take top 3
        results.sort(key=lambda x: x[1], reverse=True)
        best_points = results[:3]
        best_points.sort(key=lambda x: x[0])  # Sort by time for chronological order

        clip_paths = []
        for i, (start_time, score) in enumerate(best_points):
            output_path = f"clip_{i}.mp4"
            print(f"\nExtracting clip {i+1}/3 from {start_time:.2f}s")

            # Modified FFmpeg command to handle NAL unit errors
            cmd = (f'ffmpeg -y -hwaccel auto -i "{video_path}" -ss {start_time} -t {clip_duration} '
                  f'-c:v libx264 -preset ultrafast -crf 23 '
                  f'-c:a aac -strict experimental "{output_path}"')

            print(f"Running FFmpeg command: {cmd}")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"Created {output_path}")
                clip_paths.append(output_path)
            else:
                print(f"Error: {result.stderr}")
                # Fallback command using demuxer-level seeking
                cmd = (f'ffmpeg -y -ss {start_time} -i "{video_path}" -t {clip_duration} '
                      f'-avoid_negative_ts 1 -c copy "{output_path}"')
                print("Retrying with demuxer-level seeking...")
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"Created {output_path}")
                    clip_paths.append(output_path)
                else:
                    print(f"Error on retry: {result.stderr}")

        video.release()
        return clip_paths

    except Exception as e:
        print(f"Error: {str(e)}")
        logging.error(f"Error: {str(e)}", exc_info=True)
        raise

# ---------- SUBTITLE EXTRACTION & OVERLAY ----------

def extract_subtitles_from_srt(srt_file):
    """Extracts subtitles with timing from an SRT file."""
    subtitles = []
    current_sub = {}
    with open(srt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if '-->' in line:
            times = line.split(' --> ')
            current_sub['start'] = times[0].replace(',', '.')
            current_sub['end'] = times[1].replace(',', '.')
        elif line and not line.isdigit():
            if 'text' not in current_sub:
                current_sub['text'] = line
            else:
                current_sub['text'] += '\n' + line
        elif not line and 'text' in current_sub:
            subtitles.append(current_sub.copy())
            current_sub = {}

    return subtitles

def extract_subtitles_for_clip(subtitles, clip_start, clip_duration):
    """Extract subtitles that fall within the clip's time range."""
    clip_subtitles = []
    clip_end = clip_start + clip_duration

    for sub in subtitles:
        # Convert subtitle times to seconds
        sub_start = sum(float(x) * 60 ** i for i, x in enumerate(reversed(sub['start'].split(':'))))
        sub_end = sum(float(x) * 60 ** i for i, x in enumerate(reversed(sub['end'].split(':'))))

        # Check if subtitle overlaps with clip
        if sub_end > clip_start and sub_start < clip_end:
            # Adjust timing relative to clip start
            adjusted_sub = sub.copy()
            adjusted_sub['start'] = max(0, sub_start - clip_start)
            adjusted_sub['end'] = min(clip_duration, sub_end - clip_start)
            clip_subtitles.append(adjusted_sub)

    return clip_subtitles

def overlay_subtitles(video_file, subtitles):
    """Overlays subtitles with proper timing."""
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
        return final
    except Exception as e:
        logging.error(f"Error in overlay_subtitles: {e}")
        raise

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

        youtube = authenticate_youtube_api(api_key)

        for i, clip_path in enumerate(clip_paths):
            print(f"Processing clip {i+1}/{len(clip_paths)}...")

            if subtitle_file:
                # Calculate clip start time relative to original video
                clip_start = i * clip_duration  # Simplified; adjust based on your actual clip timing

                # Extract subtitles for this specific clip
                clip_subtitles = extract_subtitles_for_clip(all_subtitles, clip_start, clip_duration)

                if clip_subtitles:
                    print(f"Adding {len(clip_subtitles)} subtitles to clip...")
                    final_clip = overlay_subtitles(clip_path, clip_subtitles)
                    output_path = f"subtitled_clip_{i}.mp4"
                    print(f"Saving subtitled clip to {output_path}...")
                    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
                else:
                    print("No subtitles found for this clip segment")
                    output_path = clip_path
            else:
                output_path = clip_path

            # Upload the processed clip to YouTube
            transcript = transcribe_video(output_path)
            title = generate_title_with_gpt4(transcript, api_key)
            description = f"Auto-generated clip from video\nTranscript:\n{transcript}"
            upload_to_youtube(youtube, output_path, title, description)
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
