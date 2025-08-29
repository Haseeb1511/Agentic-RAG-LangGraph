# pyaudio        # Record audio from microphone.
# pydub          # Audio manipulation (convert/export to MP3, WAV, etc.).
# ffmpeg         # Required backend for pydub((needed for exporting in formats like MP3).
# SpeechRecognition   # Convert speech → text

# groq

import os
import speech_recognition as sr
from pydub import AudioSegment
from groq import Groq
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()

os.makedirs("./input_audio",exist_ok=True)
def record_audio(audio_save_path="./input_audio/input.wav"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source=source,duration=1)
        audio = recognizer.listen(source=source,timeout=10)
    
    #save raw wav
    wav_data = audio.get_wav_data()
    audio_segment = AudioSegment.from_file(BytesIO(wav_data),format="wav")
    audio_segment.export(audio_save_path,format="wav")
    audio_segment.export(audio_save_path.replace(".wav",".mp3"),format="mp3",bitrate="128k")
    return audio_save_path

def speech_to_text(audio_path:str):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    model_stt = "whisper-large-v3-turbo"
    with open(audio_path,"rb") as audio_file:
        transcription  = client.audio.transcriptions.create(
            file=audio_file,
            model=model_stt
        )
    return transcription.text
    

