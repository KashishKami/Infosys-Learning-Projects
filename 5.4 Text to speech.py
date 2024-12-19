# Import the required module for text 
# to speech conversion
from gtts import gTTS

# This module is imported so that we can 
# play the converted audio
import os

# The text that you want to convert to audio
mytext = 'तुम बस जवाब दो।  यह एक रिफ्लेक्स की तरह है। क्या मैं मोटी दिखती हूँ? नहीं। क्या वह मुझसे ज़्यादा खूबसूरत है? नहीं।'

# Language in which you want to convert
# See the langauges acronyms at: https://gtts.readthedocs.io/en/latest/module.html#languages-gtts-lang
language = 'hi'

# Passing the text and language to the engine, 
# here we have marked slow=False. Which tells 
# the module that the converted audio should 
# have a high speed
myobj = gTTS(text=mytext, lang=language, slow=False)

# Saving the converted audio in a mp3 file named
# welcome 
myobj.save("/Material/converted.mp3")

# Playing the converted file
#os.system("start welcome.mp3")