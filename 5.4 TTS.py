from transformers import pipeline
import scipy

synthesiser = pipeline("text-to-speech", model = "suno/bark", device = "cuda:0")

text = "तुम बस जवाब दो।  यह एक रिफ्लेक्स की तरह है। क्या मैं मोटी दिख रही हूँ? नहीं। क्या वह मुझसे ज़्यादा खूबसूरत है? नहीं।"

speech = synthesiser(text)

scipy.io.wavfile.write("bark_out.wav", rate=speech["sampling_rate"], data=speech["audio"])
