# pip install transformers torch

import torch
from transformers import pipeline

#To check whether the current Torch version is CUDA enabled or not
#print(torch.cuda.is_available())

# to download CUDA enabled Torch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


whisper = pipeline("automatic-speech-recognition", "openai/whisper-large-v3", torch_dtype=torch.float16, device="cuda:0")

transcription = whisper("/Material/audio1.mp3")

print(transcription["text"])



