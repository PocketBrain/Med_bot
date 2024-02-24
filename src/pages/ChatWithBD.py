import os
import importlib
import sys
import pandas as pd
import streamlit as st
from io import BytesIO
from modules.robby_sheet.table_tool import PandasAgent
from modules.layout import Layout
from modules.utils import Utilities
from modules.sidebar import Sidebar

def reload_module(module_name):

    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])
    return sys.modules[module_name]

table_tool_module = reload_module('modules.robby_sheet.table_tool')
layout_module = reload_module('modules.layout')
utils_module = reload_module('modules.utils')
sidebar_module = reload_module('modules.sidebar')


st.set_page_config(layout="wide", page_icon="💬", page_title="Мединский чат-бот 🤖")

layout, sidebar, utils = Layout(), Sidebar(), Utilities()

layout.show_header("CSV, Excel")



st.session_state.setdefault("reset_chat", False)

uploaded_file = utils.handle_upload(["csv", "xlsx"])

if uploaded_file:
    sidebar.about()
        
    uploaded_file_content = BytesIO(uploaded_file.getvalue())
    if uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" or uploaded_file.type == "application/vnd.ms-excel":
        df = pd.read_excel(uploaded_file_content)
    else:
        df = pd.read_csv(uploaded_file_content, sep=";")

    st.session_state.df = df

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    csv_agent = PandasAgent()

    with st.form(key="query"):
        query = st.text_input("Чат с CSV БД", value="", type="default",
        placeholder="e-g : How many rows ? "
        )
        submitted_query = st.form_submit_button("Submit")
        reset_chat_button = st.form_submit_button("Reset Chat")
        print(query)
    if reset_chat_button:
            st.session_state["chat_history"] = []
    if submitted_query:
        result, captured_output = csv_agent.get_agent_response(df, query)# df вместо uploaded_file
        cleaned_thoughts = csv_agent.process_agent_thoughts(captured_output)
        csv_agent.display_agent_thoughts(cleaned_thoughts)
        csv_agent.update_chat_history(query, result)
        csv_agent.display_chat_history()
    if st.session_state.df is not None:
        st.subheader("Current dataframe:")
        st.write(st.session_state.df)

