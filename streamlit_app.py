import streamlit as st
from openai import OpenAI


import os
import textwrap

import chromadb
import langchain
import openai
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader, UnstructuredPDFLoader, YoutubeLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All
from pdf2image import convert_from_path




















# Show title and description.
st.title("💬 Chatbot")
st.write(
    "This is a simple chatbot that uses OpenAI's GPT-3.5 model to generate responses. "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
    "You can also learn how to build this app step by step by [following our tutorial](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)."
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management






openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    os.environ["OPENAI_API_KEY"]

    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    model = OpenAI(temperature=0, model_name="gpt-3.5-turbo")
    images = convert_from_path("quychetext.pdf", dpi=88)
    # len(images)
    # images[0]

    # Use UnstructuredPDFLoader to load PDFs from the Internets
    pdf_loader = UnstructuredPDFLoader("quychetext.pdf")
    pdf_pages = pdf_loader.load_and_split()

    # Text Splitters
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts = text_splitter.split_documents(pdf_pages)
    # len(texts)

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    hf_embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

    db = Chroma.from_documents(texts, hf_embeddings, persist_directory="db")

    custom_prompt_template = """Sử dụng các thông tin sau đây để trả lời câu hỏi của người dùng.
    Nếu bạn không biết câu trả lời, chỉ cần nói rằng bạn không biết, đừng cố bịa ra câu trả lời.
    Tất cả câu trả lời của bạn đều phải trả lời bằng tiếng việt

    Context: {context}
    Question: {question}

    """


    from langchain import PromptTemplate
    def set_custom_prompt():
        """
        Prompt template for QA retrieval for each vectorstore
        """
        prompt = PromptTemplate(template=custom_prompt_template,
                                input_variables=['context', 'question'])
        return prompt

    prompt = set_custom_prompt()
    chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        chain_type_kwargs={'prompt': prompt}
    )

    def print_response(response: str):
        print("\n".join(textwrap.wrap(response, width=100)))

    query = "Phạm vi áp dụng có nằm trong mục đích hay không?"
    response = chain.run(query)
    # print_response(response)




    # Create a session state variable to store the chat messages. This ensures that the
    # messages persist across reruns.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display the existing chat messages via `st.chat_message`.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Create a chat input field to allow the user to enter a message. This will display
    # automatically at the bottom of the page.
    if prompt := st.chat_input("What is up?"):

        # Store and display the current prompt.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate a response using the OpenAI API.
        # stream = client.chat.completions.create(
        #     model="gpt-3.5-turbo",
        #     messages=[
        #         {"role": m["role"], "content": m["content"]}
        #         for m in st.session_state.messages
        #     ],
        #     stream=True,
        # )

        # Stream the response to the chat using `st.write_stream`, then store it in 
        # session state.
        # with st.chat_message("assistant"):
        #     response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
