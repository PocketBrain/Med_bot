import langchain
import streamlit as st
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import LlamaCpp
from llama_cpp import Llama
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain

langchain.verbose = False

class Chatbot:
    def __init__(self, model_name, temperature, vectors):
        self.model_name = model_name
        self.temperature = temperature
        self.vectors = vectors

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
        print("Init")
        n_gpu_layers = -1  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
        n_batch = 512
        llm = LlamaCpp(
            model_path="model-q8_0.gguf",
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            temperature=0.5,
            top_p=1,
            verbose=True,
            n_ctx=4096
        )
        print("how are you")
        #retriever = self.vectors.as_retriever()
        chain = LLMChain(llm=llm, prompt=self.QA_PROMPT)
        relevants = self.vectors.similarity_search(query)
        print(f"help me pls {relevants}")
        docs = relevants[0].dict()['metadata']
        chain_input = {"question": query, "context": docs['answer']}
        result = chain(chain_input)

        st.session_state["history"].append((query, result["answer"]))
        #count_tokens_chain(chain, chain_input)
        return result["answer"]


def count_tokens_chain(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        st.write(f'###### Токенов, используемых в беседе : {cb.total_tokens} tokens')
    return result 

    
    
