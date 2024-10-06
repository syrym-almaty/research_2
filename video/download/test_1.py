from pytube import YouTube
from moviepy.editor import AudioFileClip
import os

# Define the URL of the YouTube video
video_url = 'https://www.youtube.com/watch?v=g3znK6xwTlo'

# Download video in audio format (mp4)
yt = YouTube(video_url)
audio_stream = yt.streams.filter(only_audio=True).first()
download_path = audio_stream.download()

# Convert the downloaded audio (mp4) to wav format
mp4_file = download_path
wav_file = os.path.splitext(mp4_file)[0] + '.wav'

# Using moviepy to convert
audio_clip = AudioFileClip(mp4_file)
audio_clip.write_audiofile(wav_file, codec='pcm_s16le')

# Optional: Clean up the mp4 file if no longer needed
audio_clip.close()
os.remove(mp4_file)

print(f"Downloaded and converted to {wav_file}")
