# app.py

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Dict, List, Tuple
from datetime import datetime
import logging
import json
from pathlib import Path

# --- Configuration ---
class Config:
    PAGE_TITLE = "🤖 AI Assistant"
    PAGE_ICON = "🤖"
    LAYOUT = "centered"
    MAX_HISTORY = 10
    MODEL_NAME = "microsoft/DialoGPT-medium"
    MAX_LENGTH = 1000
    TEMPERATURE = 0.7

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Language Support ---
LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Hindi": "hi",
    "Arabic": "ar",
    "Russian": "ru",
    "Portuguese": "pt",
    "Italian": "it",
    "Dutch": "nl",
    "Turkish": "tr",
    "Indonesian": "id",
    "Thai": "th",
    "Vietnamese": "vi"
}

# --- AI Model ---
class AIChatbot:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Load the DialoGPT model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
            self.model = AutoModelForCausalLM.from_pretrained(Config.MODEL_NAME)
            self.model = self.model.to(self.device)
            logger.info(f"Model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            st.error("Error initializing the AI model. Please try again later.")
            st.stop()
    
    def generate_response(self, user_input: str, chat_history: List[Tuple[str, str]]) -> str:
        """Generate a response using DialoGPT model."""
        if not user_input.strip():
            return "I didn't receive any input. Could you please repeat your question?"
            
        try:
            # Format the chat history
            history = ""
            for user_msg, bot_msg in chat_history[-Config.MAX_HISTORY:]:
                history += f"User: {user_msg}\nBot: {bot_msg}\n"
            
            # Combine history with new input
            prompt = f"{history}User: {user_input}\nBot:"
            
            # Encode the input
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Generate response
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_length=len(input_ids[0]) + Config.MAX_LENGTH,
                    temperature=Config.TEMPERATURE,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode and clean the response
            response = self.tokenizer.decode(
                output[:, input_ids.shape[-1]:][0],
                skip_special_tokens=True
            )
            
            # Clean up the response
            response = response.split("User:")[0].strip()
            return response if response else "I'm not sure how to respond to that. Could you rephrase your question?"
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I encountered an error processing your request. Please try again."

# --- Session State Management ---
def initialize_session_state():
    """Initialize the session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "language" not in st.session_state:
        st.session_state.language = "English"
    if "bot" not in st.session_state:
        st.session_state.bot = AIChatbot()

def add_message(role: str, content: str):
    """Add a message to the chat history."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = {
        "role": role,
        "content": content,
        "timestamp": timestamp,
        "language": st.session_state.language
    }
    st.session_state.messages.append(message)

# --- Translation Function (Placeholder - would use a real translation API in production) ---
def translate_text(text: str, target_lang: str) -> str:
    """Translate text to the target language (placeholder implementation)."""
    # In a real app, you would use a translation API like Google Translate
    # For now, we'll just return the original text with a note
    if target_lang != "English":
        return f"[Translated to {target_lang}]: {text}"
    return text

# --- UI Components ---
def render_sidebar():
    """Render the sidebar with settings."""
    with st.sidebar:
        st.title("Settings")
        st.session_state.language = st.selectbox(
            "Select Language",
            list(LANGUAGES.keys()),
            index=list(LANGUAGES.keys()).index(st.session_state.get("language", "English"))
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("This AI assistant uses DialoGPT to provide intelligent responses.")
        st.markdown("It can answer questions on virtually any topic in multiple languages.")

def render_chat_interface():
    """Render the chat interface."""
    st.title(f"{Config.PAGE_TITLE} ({st.session_state.language})")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat
        add_message("user", prompt)
        
        # Get bot response
        bot_response = st.session_state.bot.generate_response(
            prompt,
            [(msg["content"], st.session_state.messages[i+1]["content"]) 
             for i, msg in enumerate(st.session_state.messages[:-1]) 
             if i % 2 == 0 and i+1 < len(st.session_state.messages)]
        )
        
        # Translate response if needed
        if st.session_state.language != "English":
            bot_response = translate_text(bot_response, st.session_state.language)
        
        # Add bot response to chat
        add_message("assistant", bot_response)
        
        # Rerun to update the chat
        st.rerun()

# --- Main Application ---
def main():
    # Configure page
    st.set_page_config(
        page_title=Config.PAGE_TITLE,
        page_icon=Config.PAGE_ICON,
        layout=Config.LAYOUT
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Add custom CSS
    st.markdown("""
        <style>
            .stChatMessage {
                padding: 1rem;
                border-radius: 1rem;
                margin: 0.5rem 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                line-height: 1.6;
                max-width: 85%;
            }
            .stChatMessage.user {
                background-color: #f0f4f8;
                margin-left: 15%;
                border-bottom-right-radius: 0.5rem;
            }
            .stChatMessage.assistant {
                background-color: #e3f2fd;
                margin-right: 15%;
                border-bottom-left-radius: 0.5rem;
            }
            .stChatInput {
                position: fixed;
                bottom: 2rem;
                width: 70%;
                left: 15%;
            }
            .stMarkdown {
                font-size: 1.1rem;
            }
            .stSidebar {
                background-color: #f8f9fa;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Render the UI
    render_sidebar()
    render_chat_interface()

if __name__ == "__main__":
    main()