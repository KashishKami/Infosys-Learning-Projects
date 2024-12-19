
#MP4 to MP3
from moviepy import VideoFileClip

def get_audio_from_video(video_path, audio_path):
    video = VideoFileClip(video_path)

    video.audio.write_audiofile(audio_path)



#MP3 to text
import torch
from transformers import pipeline

def get_text_from_audio(audio_path):
    whisper = pipeline("automatic-speech-recognition", "openai/whisper-large-v3", torch_dtype=torch.float16, device="cuda:0")

    transcription = whisper(audio_path)
    return transcription



#Text to Text
from dotenv import load_dotenv
import os
load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def text_translation(google_api_key, transcription, desired_lang):

    summary_prompt = """
    You are a translator where you have given an input: {input_text}
    And you need to translate this input text into {desired_language} language.
    """

    prompt_template = PromptTemplate(input_variables=["input_text", "desired_language"], template=summary_prompt)

    # Initialize the LLM with the API key
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        api_key=google_api_key
    )

    # Set up the chain
    chain = prompt_template | llm | StrOutputParser()

    # Invoke the chain
    res = chain.invoke({
        "input_text": transcription["text"],
        "desired_language": desired_lang
    })
    
    return res


# Text to Audio
from gtts import gTTS

def text_to_speech(res, new_audio_path):
    language = 'hi'
    audio = gTTS(text=res, lang=language, slow=False)
    audio.save(new_audio_path)



# Matching new audio to original
from pydub import AudioSegment
from gtts import gTTS
import os

def adjust_audio_speed(audio_file, target_duration, output_file="adjusted_audio.mp3"):
    # Handle gTTS object or file-like object
    if isinstance(audio_file, gTTS):
        temp_file = "temp_audio.mp3"
        audio_file.save(temp_file)  # Save gTTS object to a temporary file
        audio = AudioSegment.from_file(temp_file)
        os.remove(temp_file)  # Clean up the temporary file
    else:
        # Assume audio_file is a file path
        audio = AudioSegment.from_file(audio_file)

    current_duration = audio.duration_seconds
    speed_factor = current_duration / target_duration

    # Adjust speed
    adjusted_audio = audio._spawn(audio.raw_data, overrides={
        "frame_rate": int(audio.frame_rate * speed_factor)
    }).set_frame_rate(audio.frame_rate)

    adjusted_audio.export(output_file, format="mp3")
    return output_file



#Adding adjusted audio to video
from moviepy import *
def adding_audio_to_video(original_video_path, adjusted_audio, new_video_path):
    videoclip = VideoFileClip(original_video_path)
    audioclip = AudioFileClip(adjusted_audio)

    new_audioclip = CompositeAudioClip([audioclip])
    videoclip.audio = new_audioclip
    videoclip.write_videofile(new_video_path)


# def adjust_audio_speed(audio_file, original_video_path):
#     audio = AudioSegment.from_file(audio_file)
#     video = VideoFileClip(original_video_path)
#     target_duration = video.duration
#     current_duration = audio.duration_seconds
#     speed_factor = current_duration / target_duration
#     adjusted_audio = audio._spawn(audio.raw_data, overrides={
#         "frame_rate": int(audio.frame_rate * speed_factor)
#     }).set_frame_rate(audio.frame_rate)
#     #audio_stream = BytesIO()
#     return adjusted_audio

from moviepy import VideoFileClip
from pydub import AudioSegment
import tempfile

def adjust_audio_speed(audio_file, original_video_path, output_file="adjusted_audio.mp3"):
    # Load audio and video
    audio = AudioSegment.from_file(audio_file)
    video = VideoFileClip(original_video_path)
    
    # Calculate durations and speed factor
    target_duration = video.duration
    current_duration = audio.duration_seconds
    speed_factor = current_duration / target_duration

    # Adjust the audio speed
    adjusted_audio = audio._spawn(audio.raw_data, overrides={
        "frame_rate": int(audio.frame_rate * speed_factor)
    }).set_frame_rate(audio.frame_rate)

    # Save adjusted audio to a file
    adjusted_audio.export(output_file, format="mp3")
    
    # Return the path to the adjusted audio file
    return output_file



if __name__ == "__main__":

    #MP4 to MP3
    original_video_path = "C:/Users/PickleRick/Desktop/Infosys springboard/Material/video1.mp4"
    original_audio_path = "C:/Users/PickleRick/Desktop/Infosys springboard/Material/audio1.mp3"

    get_audio_from_video(original_video_path, original_audio_path)



    #Mp3 to Text
    transcription = get_text_from_audio(original_audio_path)
    print(f"Extracted text from the audio: {transcription['text']}")


    #Text to text translation
    desired_lang = input("Enter the language you want the translation: ")
    res = text_translation(google_api_key, transcription, desired_lang)
        #print(res)


    #Text to Speech
    new_audio_path = "C:/Users/PickleRick/Desktop/Infosys springboard/Material/translated.mp3"
    text_to_speech(res, new_audio_path)


    #Audio adjustment
    adjusted_audio = adjust_audio_speed(new_audio_path, original_video_path)
    
    # original_video = VideoFileClip(original_video_path)
    # original_audio_duration = original_video.audio.duration
    # adjusted_audio = adjust_audio_speed(translated_audio, original_audio_duration)


    #Adding translated audio to original video
    new_video_path = "C:/Users/PickleRick/Desktop/Infosys springboard/Material/output.mp4"

    
    adding_audio_to_video(original_video_path, adjusted_audio, new_video_path)

    