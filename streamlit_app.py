import streamlit as st
from openai import OpenAI
info = "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"
client = OpenAI(api_key=info)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant tasked with providing instrument purchase recommendations. Your role is to provide friendly advice as simply as possible. Stick to providing recommendations only, and remind users if they deviate."}
    ]

# Function to interact with ChatGPT
def chat_with_gpt(prompt, model="gpt-4"):
    try:
        completion = client.chat.completions.create(
          model="gpt-4o",
          messages=prompt,
          )
        return completion.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"

# Function to add messages and update the session state
def add_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})

# Streamlit UI
st.title("Instrument Purchase Chatbot")
st.write("Get help and advice from an AI assistant!")

# User input form
with st.form(key="chat_form"):
    user_input = st.text_input("You: ", placeholder="Type your message here...")
    submit_button = st.form_submit_button("Send")

    if submit_button and user_input.strip():
        # Add user input to chat history
        add_message("user", user_input)
        
        # Get ChatGPT's reply
        with st.spinner("AI Assistant is thinking..."):
            bot_response = chat_with_gpt(st.session_state.messages)
        
        # Add bot response to chat history
        add_message("assistant", bot_response)

# Display chat history
for msg in st.session_state.messages[1:]:  # Skip the system message
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    elif msg["role"] == "assistant":
        st.markdown(f"**AI Assistant:** {msg['content']}")

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant tasked with providing instrument purchase recommendations. Your role is to provide friendly advice as simply as possible. Stick to providing recommendations only, and remind users if they deviate."}
    ]