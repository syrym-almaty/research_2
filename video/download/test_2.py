import yt_dlp
from moviepy.editor import AudioFileClip
import os

# Define the URL of the YouTube video
# females
# video_url = 'https://www.youtube.com/watch?v=g3znK6xwTlo'

# male
# video_url = 'https://www.youtube.com/watch?v=zeApE-aD3fI'
video_url = 'https://www.youtube.com/watch?v=XLftjweg2gE'


# Set up yt-dlp options to download audio only
ydl_opts = {
    'format': 'bestaudio/best',
    'outtmpl': 'audio.mp3',  # Download as mp3 first
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
}

# Download the audio file using yt-dlp
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([video_url])

# Convert mp3 to wav using moviepy
mp3_file = 'audio.mp3'
wav_file = 'audio.wav'

audio_clip = AudioFileClip(mp3_file)
audio_clip.write_audiofile(wav_file, codec='pcm_s16le')

# Optional: Clean up the mp3 file if no longer needed
audio_clip.close()
os.remove(mp3_file)

print(f"Downloaded and converted to {wav_file}")
