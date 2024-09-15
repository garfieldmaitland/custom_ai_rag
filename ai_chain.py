# Cell 1: Install required packages
# !pip install langchain openai chromadb tiktoken unstructured textblob langchain-community gradio "unstructured[pdf]"

# Cell 2: Import required libraries
import os
import re
import sys
import openai
from textblob import TextBlob
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
import gradio as gr

# Cell 3: Set up API key and constants
# TODO: Replace 'your_api_key_here' with your actual OpenAI API key
# os.environ["OPENAI_API_KEY"] = "your_api_key_here"
os.environ["OPENAI_API_KEY"] = constants.APIKEY

# Cell 4: Define constants and settings
PERSIST = False
DATA_DIRECTORY = "Data"  # TODO: Update this to your actual data directory

# Cell 5: Create or load the index
if PERSIST and os.path.exists("persist"):
    print("Reusing index...\n")
    vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
    loader = DirectoryLoader(DATA_DIRECTORY)
    if PERSIST:
        index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
    else:
        index = VectorstoreIndexCreator(embedding=OpenAIEmbeddings()).from_loaders([loader])

# Cell 6: Create the conversational chain
chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 2}),
)

# Cell 7: Define the personalized answer function
def get_personalized_answer(query):
    # Add a personal touch to the query
    personalized_query = f"As Claude Shannon, {query}"

    result = chain({"question": personalized_query, "chat_history": []})

    # Post-process the answer to make it more personal
    answer = result['answer']
    answer = answer.replace("Claude Shannon", "you")
    answer = answer.replace("he ", "you ")
    answer = answer.replace("his ", "your ")
    answer = answer.replace("him ", "you ")

    return f"Claude, here's my suggestion: {answer}"

# Cell 8: Custom CSS for a professional theme
custom_css = """
body {
    background-color: #f5f5f5;
}
.gradio-container {
    background-color: #ffffff;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.gr-button {
    background-color: #2c3e50 !important;
    border-color: #2c3e50 !important;
}
.gr-button:hover {
    background-color: #34495e !important;
    border-color: #34495e !important;
}
.gr-input {
    border-color: #bdc3c7 !important;
}
footer {
    display: none !important;
}
"""

# Cell 9: Create and launch Gradio interface
iface = gr.Interface(
    fn=get_personalized_answer,
    inputs=gr.Textbox(lines=2, placeholder="Ask me anything about optimizing your life, Claude..."),
    outputs="text",
    title="Claude's Personal AI Assistant",
    description="Your personal AI assistant to help optimize your life and achieve your goals.\r\n (powered by GPT-3.5 Turbo)",
    examples=[
        ["If you were me, how would you optimize my career?"],
        ["What strategies can I use to improve my work-life balance?"],
        ["How can I leverage my strengths in my career?"],
        ["What are some areas for personal growth I should focus on?"],
        ["How can I harness machine learning, to augment and improve my career?"],
    ],
    css=custom_css,
    theme=gr.themes.Soft()
)

iface.launch(share=True, show_api=False)
