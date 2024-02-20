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
- **Медицинский робот**: Отвечает на вопросы исходя из контекста медицинских данных
- **Медицинский робот для БД**  Отвечает на основываясь на данных о пациентах из БД
""")
st.markdown("---")






