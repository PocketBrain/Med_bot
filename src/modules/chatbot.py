import langchain
import streamlit as st
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import LlamaCpp
from llama_cpp import Llama
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
import pickle
langchain.verbose = False

class Chatbot:
    def __init__(self, vectors, file, uploaded_file):
        self.vectors = vectors
        self.PATH = "embeddings"
        self.file = file
        self.uploaded_file = uploaded_file
        self.llm = LlamaCpp(
            model_path="model-q8_0.gguf",
            n_gpu_layers=-1,
            n_batch=512,
            temperature=0.5,
            top_p=1,
            verbose=True,
            n_ctx=4096
        )

    qa_template = """
        Вы - Мединский робот, полезный ассистент. Пользователь предоставляет вам файл, содержимое которого представлено следующими фрагментами контекста, используйте их, чтобы ответить на вопрос в конце.
 Если вы не знаете ответа, просто скажите, что не знаете. НЕ пытайтесь придумать ответ.
 Если вопрос не связан с контекстом, вежливо ответьте, что знаете, исходя из собственных знаний.
 При ответе используйте как можно больше деталей.

        Контекст: {context}
        =========
        Вопрос: {question}
        ======
        """

    QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["context","question" ])

    def conversational_chat(self, query):
        chain = LLMChain(llm=self.llm, prompt=self.QA_PROMPT)
        relevants = self.vectors.similarity_search(query)
        print(f"help me pls {relevants}")
        docs = relevants[0].dict()['metadata']
        chain_input = {"question": query, "context": docs['answer']}
        result = chain(chain_input)

        st.session_state["history"].append((query, result["answer"]))
        # count_tokens_chain(chain, chain_input)
        return result["answer"]


    def count_tokens_chain(chain, query):
        result = chain.invoke(query)
        return result

    
    
