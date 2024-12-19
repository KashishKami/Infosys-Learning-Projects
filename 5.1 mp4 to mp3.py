from moviepy import VideoFileClip

# Load the video file
video = VideoFileClip("/Material/video1.mp4")

# Extract and save the audio as MP3
video.audio.write_audiofile("/Material/audio1.mp3")