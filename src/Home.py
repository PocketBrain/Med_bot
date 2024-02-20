import streamlit as st


#Config
st.set_page_config(layout="wide", page_icon="💬", page_title="Медицинский чат бот")


#Title
st.markdown(
    """
    <h2 style='text-align: center;'>Мединский робот - ваш личный ассистент 🤖</h1>
    """,
    unsafe_allow_html=True,)

st.markdown("---")


#Description
st.markdown(
    """ 
    <h5 style='text-align:center;'>Я Мед Робот, интеллектуальный чат-бот, созданный путем объединения
     Langchain и MistralAI. Я использую большие языковые модели для обеспечения
контекстно-зависимых взаимодействий. Моя цель - помочь вам лучше понять ваши данные.
 Я поддерживаю PDF, TXT, CSV,  🧠</h5>
    """,
    unsafe_allow_html=True)
st.markdown("---")


#Robby's Pages
st.subheader("🚀 Страницы робота")
st.write("""
- **Медицинский робот**: General Chat on data (PDF, TXT,CSV) with a [vectorstore](https://github.com/facebookresearch/faiss) (index useful parts(max 4) for respond to the user) | works with [ConversationalRetrievalChain](https://python.langchain.com/en/latest/modules/chains/index_examples/chat_vector_db.html)
- **Медицинский робот** (beta): Chat on tabular data (CSV) | for precise information | process the whole file | works with [CSV_Agent](https://python.langchain.com/en/latest/modules/agents/toolkits/examples/csv.html) + [PandasAI](https://github.com/gventuri/pandas-ai) for data manipulation and graph creation""")
st.markdown("---")






