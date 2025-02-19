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

# ---------- VIDEO SPLITTING ----------

def split_video_fixed_duration(video_path, clip_duration):
    """
    Splits the video into clips of fixed duration.
    Uses FFmpeg via system command.
    """
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_length_frames = int(clip_duration * fps)
    num_clips = frame_count // clip_length_frames

    clip_paths = []
    for i in range(num_clips):
        start_time = i * clip_duration
        end_time = (i + 1) * clip_duration
        output_path = f"clip_{i}.mp4"
        # Extract clip using FFmpeg
        cmd = f"ffmpeg -y -i \"{video_path}\" -ss {start_time} -to {end_time} -c copy \"{output_path}\""
        subprocess.call(cmd, shell=True)
        clip_paths.append(output_path)
    video.release()
    return clip_paths

def split_video_scene_detection(video_path, threshold=30):
    """
    Splits the video into scenes using PySceneDetect.
    Note: This is a simplified example.
    """
    # Detect scenes using PySceneDetect's command line interface or API
    # For simplicity, we assume a function 'detect_scenes' returns a list of (start, end) times in seconds.
    scene_list = []  # Replace with actual scene detection code or API call
    # For demonstration, let’s assume one scene covering the entire video:
    video = mpy.VideoFileClip(video_path)
    scene_list.append((0, video.duration))
    clip_paths = []
    for i, (start_time, end_time) in enumerate(scene_list):
        output_path = f"scene_{i}.mp4"
        cmd = f"ffmpeg -y -i \"{video_path}\" -ss {start_time} -to {end_time} -c copy \"{output_path}\""
        subprocess.call(cmd, shell=True)
        clip_paths.append(output_path)
    return clip_paths

# ---------- SUBTITLE EXTRACTION & OVERLAY ----------

def extract_subtitles_from_srt(srt_file):
    """Extracts subtitles from an SRT file."""
    with open(srt_file, 'r', encoding='utf-8') as f:
        subtitles = f.read()
    return subtitles

def overlay_subtitles(video_file, subtitles, font='Arial', fontsize=24, color='white', position='bottom', animation=False):
    """Overlays subtitles on the video clip using MoviePy."""
    video = mpy.VideoFileClip(video_file)
    # Create a TextClip for subtitles (this example places the entire subtitle text over the video)
    text = mpy.TextClip(subtitles, font=font, fontsize=fontsize, color=color, bg_color='black')
    text = text.set_pos(position).set_duration(video.duration)

    # Optional animated effects (placeholder - replace with actual effect if needed)
    if animation:
        # e.g., text_clip = text_clip.fx(your_animation_effect)
        pass

    final = mpy.CompositeVideoClip([video, text])
    return final

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
    logging.basicConfig(filename='video_processing.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        # Check for FFmpeg first
        if not check_ffmpeg():
            return

        # Step 1: Split video
        if clip_duration > 0:
            print("Splitting video by fixed duration...")
            clip_paths = split_video_fixed_duration(video_path, clip_duration)
        else:
            print("Splitting video by scene detection...")
            clip_paths = split_video_scene_detection(video_path)

        # Step 2: Authenticate YouTube API
        youtube = authenticate_youtube_api(api_key)

        # Process each clip
        for clip_path in clip_paths:
            output_path = clip_path
            # Step 3: Process subtitles if provided
            if subtitle_file:
                print(f"Overlaying subtitles on {clip_path}...")
                subtitles = extract_subtitles_from_srt(subtitle_file)
                final_clip = overlay_subtitles(clip_path, subtitles)
                output_path = "subtitled_" + clip_path
                final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

            # Step 4: Generate title
            print(f"Transcribing {clip_path} for title generation...")
            transcript = transcribe_video(clip_path)
            title = generate_title_with_gpt4(transcript, api_key)
            print(f"Generated title: {title}")

            # Step 5: Upload to YouTube
            print(f"Uploading {output_path} to YouTube...")
            upload_to_youtube(youtube, output_path, title, "Short clip from original video")
            logging.info(f"Processed and uploaded {clip_path} with title: {title}")
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
