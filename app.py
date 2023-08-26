from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st

"""
# Всем привет

Пока эта приложуха просто отвечает вашим же сообщением
"""

msg = st.text_area('Ваше сообщение:', placeholder='бла-бла-бла')

if st.button('Отправить'):
    f"""
    ### Ответное сообщение:
    {msg}
    """




