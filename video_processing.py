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

print(os.environ['PATH'])

# ---------- VIDEO SPLITTING ----------

def is_dark_frame(frame, threshold=30):
    """Check if frame is too dark based on average pixel intensity."""
    return cv2.mean(frame)[0] < threshold

def analyze_scene_content(video_path, start_time, duration):
    """Analyze scene content for interesting moments."""
    video = cv2.VideoCapture(video_path)
    video.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

    frame_scores = []
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_to_check = int(duration * fps)

    for _ in range(frames_to_check):
        ret, frame = video.read()
        if not ret:
            break

        # Calculate scene score based on multiple factors
        motion_score = cv2.norm(cv2.absdiff(frame, prev_frame)) if 'prev_frame' in locals() else 0
        brightness_score = cv2.mean(frame)[0]
        contrast_score = frame.std()

        frame_scores.append(motion_score + brightness_score + contrast_score)
        prev_frame = frame

    video.release()
    return sum(frame_scores) / len(frame_scores) if frame_scores else 0

def split_video_fixed_duration(video_path, clip_duration):
    """Split video into 3 random clips."""
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_length_frames = int(clip_duration * fps)

    # Skip dark intro
    dark_frames = 0
    while True:
        ret, frame = video.read()
        if not ret or not is_dark_frame(frame):
            break
        dark_frames += 1

    start_time = dark_frames / fps
    print(f"Skipped {start_time:.2f} seconds of dark intro")

    # Calculate remaining duration
    remaining_duration = (frame_count - dark_frames) / fps

    # Set number of clips to 3
    num_clips = 3

    # Generate 3 random start times (ensuring clips don't overlap)
    import random
    potential_clips = []
    for i in range(num_clips):
        clip_start = start_time + (i * clip_duration)
        score = analyze_scene_content(video_path, clip_start, clip_duration)
        potential_clips.append((clip_start, score))

    if len(potential_clips) < 3:
        print("Warning: Video too short for 3 clips")
        random_starts = [clip[0] for clip in potential_clips]
    else:
        random_starts = [clip[0] for clip in sorted(potential_clips, key=lambda x: x[1], reverse=True)[:3]]
    random_starts.sort()  # Keep chronological order

    # Generate clips
    clip_paths = []
    for i, clip_start in enumerate(random_starts):
        clip_end = clip_start + clip_duration
        output_path = f"clip_{i}.mp4"
        cmd = f'ffmpeg -y -i "{video_path}" -ss {clip_start} -to {clip_end} -c copy "{output_path}"'
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

def overlay_subtitles(video_file, subtitles):
    """Overlays subtitles with proper timing."""
    video = mpy.VideoFileClip(video_file)
    subtitle_clips = []

    for sub in subtitles:
        start_time = sum(float(x) * 60 ** i for i, x in enumerate(reversed(sub['start'].split(':'))))
        end_time = sum(float(x) * 60 ** i for i, x in enumerate(reversed(sub['end'].split(':'))))

        text_clip = mpy.TextClip(sub['text'],
                                font='Arial',
                                fontsize=24,
                                color='white',
                                bg_color='black',
                                size=(video.w, None),
                                method='caption')
        text_clip = text_clip.set_position(('center', 'bottom'))
        text_clip = text_clip.set_start(start_time).set_end(end_time)
        subtitle_clips.append(text_clip)

    final = mpy.CompositeVideoClip([video] + subtitle_clips)
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
    try:
        if not check_ffmpeg():
            return

        print("Analyzing video content and splitting into interesting clips...")
        clip_paths = split_video_fixed_duration(video_path, clip_duration)

        youtube = authenticate_youtube_api(api_key)

        for clip_path in clip_paths:
            if subtitle_file:
                print(f"Processing subtitles for {clip_path}...")
                subtitles = extract_subtitles_from_srt(subtitle_file)
                final_clip = overlay_subtitles(clip_path, subtitles)
                output_path = "subtitled_" + clip_path
                final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
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
