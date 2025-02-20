#!/usr/bin/env python3
import os
import cv2
import logging
import argparse
import subprocess
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import openai
import numpy as np
from multiprocessing import Pool

# Remove unused imports
# import scenedetect
# import moviepy.editor as mpy
# from moviepy.editor import VideoFileClip


# Two blank lines before functions
def is_dark_frame(frame, threshold=30):
    """Check if frame is too dark based on average pixel intensity."""
    return cv2.mean(frame)[0] < threshold


def analyze_scene_content(args):
    """Analyze scene content with minimal processing."""
    video_path, start_time, duration = args
    try:
        video = cv2.VideoCapture(video_path)
        video.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

        frames_to_check = 5
        frame_scores = []

        for _ in range(frames_to_check):
            ret, frame = video.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            edges = cv2.Canny(gray, 100, 200)
            edge_score = np.mean(edges)

            score = brightness * 0.4 + edge_score * 0.6
            frame_scores.append(score)

            video.set(cv2.CAP_PROP_POS_FRAMES,
                     video.get(cv2.CAP_PROP_POS_FRAMES) + 30)  # Fix indentation

        video.release()
        return (start_time, np.mean(frame_scores) if frame_scores else 0)
    except Exception as e:
        logging.error(f"Error in analyze_scene_content: {e}")
        return (start_time, 0)


def split_video_fixed_duration(video_path, clip_duration):
    """Split video into 50 clips with high quality."""
    try:
        print("Opening video file...")
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise Exception("Could not open video file")

        total_duration = (video.get(cv2.CAP_PROP_FRAME_COUNT) /
                        video.get(cv2.CAP_PROP_FPS))
        print(f"Video duration: {total_duration:.2f} seconds")

        start_offset = 60
        end_offset = 60
        sample_points = [
            (video_path, time, clip_duration)
            for time in range(
                int(start_offset),
                int(total_duration - end_offset - clip_duration),
                30
            )
        ]

        print("Analyzing scenes (this may take a few minutes)...")
        with Pool(processes=4) as pool:
            results = pool.map(analyze_scene_content, sample_points)

        results.sort(key=lambda x: x[1], reverse=True)
        best_points = results[:50]
        best_points.sort(key=lambda x: x[0])

        clip_paths = []
        for i, (start_time, score) in enumerate(best_points):
            output_path = f"clip_{i}.mp4"
            print(f"Extracting clip {i+1}/50 from {start_time:.2f}s")

            cmd = (
                f'ffmpeg -y -ss {start_time} -i "{video_path}" -t {clip_duration} '
                f'-c:v libx264 -preset medium -crf 18 -profile:v high '
                f'-c:a aac -b:a 192k "{output_path}"'
            )

            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                clip_paths.append(output_path)
            else:
                logging.error(f"FFmpeg error: {result.stderr}")

        video.release()
        return clip_paths

    except Exception as e:
        logging.error(f"Error in split_video_fixed_duration: {e}", exc_info=True)
        raise


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


def transcribe_video(video_file):
    """
    Dummy transcription function.
    Replace this with a call to an actual transcription service (e.g., OpenAI Whisper).
    """
    # For demonstration, we return a fixed string.
    return "This is a sample transcript of the video content."


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


def main(video_path, clip_duration, api_key):
    """Orchestrates the video processing pipeline."""
    try:
        if not check_ffmpeg():
            return

        print("Splitting video into clips...")
        clip_paths = split_video_fixed_duration(video_path, clip_duration)

        processed_clips = clip_paths

        # Upload clips to YouTube
        youtube = authenticate_youtube_api(api_key)
        for i, clip_path in enumerate(processed_clips):
            print(f"\nUploading clip {i+1}/{len(processed_clips)}...")
            transcript = transcribe_video(clip_path)
            title = generate_title_with_gpt4(transcript, api_key)
            description = f"Auto-generated clip from video\nTranscript:\n{transcript}"
            upload_to_youtube(youtube, clip_path, title, description)

        # Clean up temporary files
        for path in clip_paths:
            try:
                os.remove(path)
                print(f"Cleaned up {path}")
            except Exception:  # Specify the exception type
                logging.error(f"Failed to clean up {path}", exc_info=True)

    except Exception:
        logging.error("An error occurred.", exc_info=True)
        print("An error occurred. Check the log for details.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automated Video Processing and YouTube Upload"
    )
    parser.add_argument(
        "--video_path",
        required=True,
        help="Path to the input video file"
    )
    parser.add_argument(
        "--clip_duration",
        type=int,
        default=30,
        help="Duration (in seconds) for fixed clip splitting"
    )
    parser.add_argument(
        "--api_key",
        required=True,
        help="API key for GPT-4 title generation and YouTube upload"
    )

    args = parser.parse_args()
    main(args.video_path, args.clip_duration, args.api_key)
