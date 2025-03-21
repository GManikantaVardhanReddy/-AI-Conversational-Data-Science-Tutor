import streamlit as st
import google.generativeai as genai
from langchain.memory import ConversationBufferMemory
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Google Gemini API
try:
    genai.configure(api_key="AIzaSyDplai8TdzRpuYaY73_cRD0JiZajCyhqu4")
    model = genai.GenerativeModel(
        "gemini-1.5-pro",
        generation_config={
            "temperature": 0.7,
            "max_output_tokens": 1000
        }
    )
except Exception as e:
    st.error(f"Failed to initialize Gemini model: {str(e)}")
    logger.error(f"Model initialization error: {str(e)}")
    st.stop()

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Streamlit UI
st.set_page_config(page_title="AI Data Science Tutor", layout="wide")

st.title("🤖 AI Data Science Tutor")
st.markdown("Ask any data science question and get expert answers!")

# Sidebar Controls
with st.sidebar:
    st.header("Controls")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        memory.clear()
        st.success("Chat history cleared!")

# Initialize session state for chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Function to get response from Gemini
def get_response(question):
    try:
        # Retrieve chat history from session state
        chat_history = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-5:]]
        )

        prompt = f"""You are an expert Data Science Tutor. Provide clear, accurate, and concise answers.
        Previous conversation:
        {chat_history}

        User question: {question}"""

        response = model.generate_content(prompt)

        # Ensure response is in text format
        if hasattr(response, "text"):
            return response.text
        else:
            return response.result

    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return f"Sorry, I encountered an error: {str(e)}"

# User input
user_input = st.chat_input("Ask a data science question...")

if user_input:
    try:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Get and display response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_response(user_input)
                st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Rerun Streamlit app to update UI
        st.rerun()

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Chat error: {str(e)}")

# Styling
st.markdown(
    """
    <style>
    .stChatMessage {
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)
