import re
import sys
from io import StringIO, BytesIO
import matplotlib.pyplot as plt
import streamlit as st
from langchain.callbacks import get_openai_callback
from streamlit_chat import message

from pandasai import PandasAI
from langchain_community.llms import LlamaCpp

class PandasAgent :

    @staticmethod
    def count_tokens_agent(agent, query):

        with get_openai_callback() as cb:
            result = agent(query)
            st.write(f'Spent a total of {cb.total_tokens} tokens')

        return result
    
    def __init__(self):
        pass

    def get_agent_response(self, uploaded_file_content, query):
        n_gpu_layers = -1  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
        n_batch = 512
        llm = LlamaCpp(
            model_path="model-q8_0.gguf",
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            n_threads=16,
            temperature=0.5,
            top_p=1,
            verbose=True,
            n_ctx=4096
        )
        pandas_ai = PandasAI(llm)
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        response = pandas_ai.run(data_frame = uploaded_file_content, prompt=query)
        fig = plt.gcf()
        if fig.get_axes():
                    # Adjust the figure size
            fig.set_size_inches(12, 6)

            # Adjust the layout tightness
            plt.tight_layout()
            buf = BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            st.image(buf, caption="Generated Plot")
        
        sys.stdout = old_stdout
        return response, captured_output

    def process_agent_thoughts(self,captured_output):
        thoughts = captured_output.getvalue()
        cleaned_thoughts = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', thoughts)
        cleaned_thoughts = re.sub(r'\[1m>', '', cleaned_thoughts)
        return cleaned_thoughts

    def display_agent_thoughts(self,cleaned_thoughts):
        with st.expander("Display the agent's thoughts"):
            st.write(cleaned_thoughts)

    def update_chat_history(self,query, result):
        st.session_state.chat_history.append(("user", query))
        st.session_state.chat_history.append(("agent", result))

    def display_chat_history(self):
        for i, (sender, message_text) in enumerate(st.session_state.chat_history):
            if sender == "user":
                message(message_text, is_user=True, key=f"{i}_user")
            else:
                message(message_text, key=f"{i}")