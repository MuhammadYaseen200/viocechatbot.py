# Install necessary packages
!pip install openai-whisper gtts gradio groq
import os
os.environ["GROQ_API_KEY"] = "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"  # Replace with your actual API key
# Import libraries
import whisper
import os
from gtts import gTTS
import gradio as gr
from groq import Groq
import warnings

# Suppress the FutureWarning from PyTorch
warnings.filterwarnings("ignore", category=FutureWarning)

# Set your Groq API key
os.environ["GROQ_API_KEY"] = "your-groq-api-key-here"  # Replace with your actual API key

# Load the Whisper model (small model for faster inference)
model = whisper.load_model("small", download_root=".")

# Initialize Groq API client with your API key
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Step 1: Function to transcribe audio input using Whisper
def transcribe_audio(file_path):
    # Transcribe the audio file to text
    result = model.transcribe(file_path)
    return result['text']

# Step 2: Function to send the transcribed text to the Groq LLM
def query_llm(text):
    # Call Groq API to get a completion from the LLM (Llama 3)
    chat_completion = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": text,
        }],
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content

# Step 3: Function to convert LLM response text to audio using gTTS
def text_to_speech(text, filename="response.mp3"):
    # Convert the LLM response text to speech
    tts = gTTS(text=text, lang='en')
    tts.save(filename)
    return filename

# Step 4: Combine transcription, LLM query, and speech synthesis in a single function
def chatbot(audio_file):
    # Transcribe the audio input
    transcription = transcribe_audio(audio_file)

    # Send transcribed text to the LLM for a response
    llm_response = query_llm(transcription)

    # Convert LLM response text to speech
    audio_output = text_to_speech(llm_response)

    # Return the transcription, LLM response, and audio output file
    return transcription, llm_response, audio_output

# Step 5: Create a Gradio interface for easy user interaction
interface = gr.Interface(
    fn=chatbot,  # The chatbot function
    inputs=gr.Audio(type="filepath"),  # Input audio from microphone or file
    outputs=[  # Outputs for transcription, LLM response text, and audio response
        "text",  # Transcribed text
        "text",  # LLM response
        "audio"  # Generated audio response
    ],
    title="Real-Time Voice-to-Voice Chatbot",  # Title of the Gradio interface
    description="Speak to the chatbot and get a voice response in real-time."  # Description for users
)

# Step 6: Launch the Gradio interface
interface.launch()

