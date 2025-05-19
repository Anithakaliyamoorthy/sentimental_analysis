import streamlit as st
from streamlit_audio_recorder import audio_recorder
import speech_recognition as sr
import io
from pydub import AudioSegment

# Set Streamlit page config as the FIRST command
st.set_page_config(page_title="🎙️ Voice Recorder & Transcriber", layout="centered")

st.title("🎙️ Voice Recorder & Transcriber")
st.markdown("Record your voice and transcribe it using Google's Speech Recognition.")

# Record audio
audio_bytes = audio_recorder(text="Click to record", recording_color="#e53935", neutral_color="#6c757d", icon_name="microphone")

# Display player
if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    st.info("Transcribing... Please wait.")

    try:
        # Convert bytes to WAV for recognizer
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)

        # Transcribe using Google
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_io) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            st.success("✅ Transcription:")
            st.write(f"🗣️ **{text}**")

    except sr.UnknownValueError:
        st.error("Google could not understand the audio.")
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")


