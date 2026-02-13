## Development of a PDF-Based Question-Answering Chatbot Using LangChain 
## Name: MARIMUTHU MATHAVAN
## Register no: 212224230153
### AIM :
To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

### PROBLEM STATEMENT :
The objective is to build a chatbot that answers questions from the content of a PDF document. Using LangChain and OpenAI, the system processes the PDF, retrieves relevant information, and provides accurate responses to user queries.

### DESIGN STEPS :
#### STEP 1 :
Load and Process PDF – Import the PDF document, extract its content, and split it into smaller text chunks for efficient handling.
#### STEP 2 :
Embed and Store Content – Convert text chunks into embeddings and store them in an in-memory vector database for fast retrieval.
#### STEP 3 :
Build Question-Answering System – Use LangChain’s RetrievalQA with an OpenAI model to retrieve relevant chunks and generate accurate answers to user queries.

### PROGRAM :
```py

import os, datetime, sys
import panel as pn, param
from dotenv import load_dotenv, find_dotenv
from langchain.vectorstores import Chroma, DocArrayInMemorySearch
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
pn.extension()
_ = load_dotenv(find_dotenv())
os.environ["OPENAI_API_KEY"]
llm_name = "gpt-3.5-turbo" if datetime.datetime.now().date() >= datetime.date(2023, 9, 2) else "gpt-3.5-turbo-0301"
def load_db(file, chain_type="stuff", k=4):
    docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(PyPDFLoader(file).load())
    db = DocArrayInMemorySearch.from_documents(docs, OpenAIEmbeddings())
    retriever = db.as_retriever(search_kwargs={"k": k})
    return ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=0),
        retriever=retriever,
        chain_type=chain_type,
        return_source_documents=True,
        return_generated_question=True,
    )
class ChatBot(param.Parameterized):
    chat_history, db_response = param.List([]), param.List([])
    answer, db_query = param.String(""), param.String("")

    def __init__(self, **params):
        super().__init__(**params)
        self.loaded_file, self.panels = "docs/cs229_lectures/pdf-3.pdf", []
        self.qa = load_db(self.loaded_file)

    def call_load_db(self, count):
        if file_input.value:
            file_input.save("temp.pdf")
            self.loaded_file = file_input.filename
            self.qa, button_load.button_style = load_db("temp.pdf"), "solid"
        else:
            button_load.button_style = "outline"
        self.clr_history()
        return pn.pane.Markdown(f"Loaded File: {self.loaded_file}")

    def convchain(self, query):
        if not query: return pn.WidgetBox(pn.Row('User:', pn.pane.Markdown("")), scroll=True)
        result = self.qa({"question": query, "chat_history": self.chat_history})
        self.chat_history.append((query, result["answer"]))
        self.db_query, self.db_response, self.answer = result["generated_question"], result["source_documents"], result["answer"]
        self.panels.extend([
            pn.Row('User:', pn.pane.Markdown(query, width=600)),
            pn.Row('ChatBot:', pn.pane.Markdown(self.answer, width=600, style={'background-color': '#F6F6F6'}))
        ])
        inp.value = ''
        return pn.WidgetBox(*self.panels, scroll=True)

    def get_lquest(self): return pn.pane.Str(self.db_query or "no DB accesses so far")
    def get_sources(self): return pn.WidgetBox(*[pn.Row(pn.pane.Str(doc)) for doc in self.db_response]) if self.db_response else None
    def get_chats(self): return pn.WidgetBox(*[pn.Row(pn.pane.Str(x)) for x in self.chat_history]) if self.chat_history else pn.pane.Str("No History Yet")
    def clr_history(self, *_): self.chat_history, self.panels = [], []
cb = ChatBot()
file_input, button_load, button_clearhistory = pn.widgets.FileInput(accept='.pdf'), pn.widgets.Button(name="Load DB"), pn.widgets.Button(name="Clear History", button_type='warning')
button_clearhistory.on_click(cb.clr_history)
inp = pn.widgets.TextInput(placeholder='Enter text here…')
bound_button_load, conversation = pn.bind(cb.call_load_db, button_load.param.clicks), pn.bind(cb.convchain, inp)
tab1 = pn.Column(inp, pn.layout.Divider(), pn.panel(conversation, loading_indicator=True, height=300))
tab2, tab3, tab4 = pn.Column(cb.get_lquest), pn.Column(cb.get_chats), pn.Column(file_input, button_load, bound_button_load, button_clearhistory)
dashboard = pn.Column(pn.pane.Markdown('# ChatWithYourData_Bot'), pn.Tabs(('Conversation', tab1), ('Database', tab2), ('Chat History', tab3), ('Configure', tab4)))
dashboard

```
### OUTPUT :
<img width="1036" height="584" alt="image" src="https://github.com/user-attachments/assets/190c8e86-a9a7-44ab-9a38-ceb6c6930f1b" />
<img width="942" height="577" alt="image" src="https://github.com/user-attachments/assets/539f80de-3403-4b4d-a9ec-6852d5aef003" />
<img width="872" height="159" alt="image" src="https://github.com/user-attachments/assets/185bedc9-e34b-43bb-9d1c-0bb8a944dfe3" />

### RESULT :
Thus, a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain is executed successfully.
