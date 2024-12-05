import streamlit as st
from openai import OpenAI
client = OpenAI(api_key='sk-EHS-Dx_gbEApNTp85UGREsJV4GbcGZGlkBYy_RMTxoT3BlbkFJoQuLo9dMhcyh-BFUvtgrgnQLUkdEBxHtovpJ2CuD8A')

# Function to interact with ChatGPT
def chat_with_gpt(prompt, model="gpt-4"):
    try:
        completion = client.chat.completions.create(
          model="gpt-4o",
          messages=[
          {"role": "system", "content": "You are a helpful assistant tasked with providing instrument purchase recommendations. Your role is to provide friendly advice as simply as possible. Stick to providing recommendations only, and remind users if they deviate."},
          {
            "role": "user",
            "content": prompt,
          }
         ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"

# Streamlit UI
st.title("Instrument Purchase Chatbot")
st.write("Get help and advice from an AI assistant!")

# User input
user_input = st.text_input("You: ", placeholder="Type your message here...")

if st.button("Send"):
    if user_input.strip():
        with st.spinner("Thinking..."):
            bot_response = chat_with_gpt(user_input)
        st.write(f"**AI Assistant**: {bot_response}")
    else:
        st.warning("Please type something to chat!")

# Optional: Clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []