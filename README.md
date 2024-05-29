
# Semantic Similarity and Conversational Interface

I have developed a streamlit application to perform all of the following functions. The app accepts text files that can be uploaded. The content of the text files is extracted and embedded using OpenAI embeddings. These embeddings along with their metadata such as document name and chunk ID are saved into a Weaviate vector database. Once uploaded, the unique list of documents is displayed in the sidebar. The sidebar also has a search bar where you can enter text. On hitting enter, it performs a top K (k=3) semantic search  on the chunks returning the most similar chunks with their document name and chunk ID.
**Important Note - All functionalities of the app perform only once at least one text file has been uploaded.** On the main page, there is a chat option that utilizes OpenAI GPT 3.5 Turbo through langchain to converse with the user. On the chat bar, you can ask any relevant questions regarding the uploaded documents and it will answer. In addition to implementing the RAG pipeline, chat history has also been implemented to make the chatting experience much more smoother. With chat history, the model takes into consideration the previous messages to answer the current query more suitably.
 
 
### Prerequisites

A list of dependencies
```
langchain==0.1.16
langchain-community==0.0.34
langchain-core==0.1.46
streamlit==1.33.0
streamlit-extras==0.4.0
weaviate-client==3.26.0
tiktoken==0.5.1
openai==0.28.1
```

## Steps to run

1. Clone the GitHub Repository
```bash
$ git clone https://github.com/Shaashwat05/conversationalAI
```
2. Install all dependencies using pip
```bash
$ pip install requirements.txt
```
3. Cd to the current cloned directory. In the app.py python file, enter a suitable API key for OpenAI models. Run the following command to start the application:
```bash
$ streamlit run app.py
```

