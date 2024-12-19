video_path = "/Material/video1.mp4"
audio_path = "/Material/converted.mp3"


from moviepy import *
videoclip = VideoFileClip(video_path)
audioclip = AudioFileClip(audio_path)

new_audioclip = CompositeAudioClip([audioclip])
videoclip.audio = new_audioclip
videoclip.write_videofile("/Material/output.mp4")


