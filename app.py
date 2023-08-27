import datetime
import os

import pandas as pd
import streamlit as st
from PIL import Image

from script import skills_retrieve, EMP_PATH, match_employees, sentiment


def test_match_employees():
    return pd.DataFrame(["Employee from Russia bla-bla-bla", "Employee from Poland bla-bla-bla"], index=["Yury S", "Andrei P"], columns=["evaluation"])

"""
# AI Manager Tool

This app matches employees to the project.
"""
with st.form('Match employees'):
    "### Describe your project"

    name = st.text_input('Input project name:', placeholder='Personalized Content Recommendation Engine for Online Streaming Platform')
    overview = st.text_area('Input project overview', placeholder="""In this project description, we present the details of an IT-company project focused on leveraging Machine Learning (ML) and Data Science to develop an innovative recommendation engine for an online streaming platform. The project aims to enhance user experience, increase engagement, and optimize content recommendations based on individual preferences.
    """)
    complete_by = st.date_input('Choose project completion date')
    goals = st.text_area('Input project goals', placeholder="""The primary objective of this IT-company project is to design, build, and deploy a sophisticated recommendation engine that utilizes Machine Learning and Data Science techniques to:
    1. Enhance User Engagement: Provide users with tailored content recommendations that match their viewing history, preferences, and behaviors.
    2. Improve Content Discovery: Facilitate the discovery of new content by suggesting relevant movies, TV shows, and genres that align with users' interests.
    3. Optimize Viewing Experience: Increase user satisfaction by reducing the time spent searching for content and improving the relevance of recommendations.
    4. Boost Platform Retention: Encourage users to spend more time on the platform by consistently delivering compelling and personalized content.
    5. Drive Business Value: Translate improved user engagement into higher viewer retention rates, increased subscriptions, and enhanced brand loyalty.
    """)
    skillset = st.multiselect('Choose required skills', skills_retrieve(EMP_PATH))
    desired_outcomes = st.text_area('Input project desired outcomes', placeholder="""The successful completion of this project will result in a cutting-edge recommendation engine integrated into the online streaming platform. The engine will deliver accurate and personalized content suggestions to users, ultimately enhancing their viewing experience, increasing engagement, and contributing to the platform's business success.
    """)
    match_button = st.form_submit_button('Match employees to this project!')

if match_button:
    PAT = os.environ.get('CLARIFAI_PAT')
    if not PAT:  # If PAT is not set via environment variable
        try:
            PAT = st.secrets['CLARIFAI_PAT']
        except KeyError:
            st.error("Failed to retrieve the Clarifai Personal Access Token!")
            PAT = None

    model_id = os.environ.get('MODEL_ID')
    if not model_id:
        try:
            model_id = st.secrets['MODEL_ID']
        except KeyError:
            st.error("Failed to retrieve model id!")
            model_id = None

    # Рассчитываем время на выполнение проекта
    duration = str(complete_by - datetime.date.today())

    # Формируем строку скиллов
    skillset = ', '.join(skillset)

    _, _, evaluation = match_employees(PAT, model_id, name, overview, duration, goals, skillset, desired_outcomes)
    #evaluation = test_match_employees()
    positive = []
    value = []
    for eval in evaluation['evaluation']:
        try:
            prob = sentiment(PAT, str(eval))['POSITIVE']
            value.append(prob)
            positive.append(prob > 0.5)
        except:
            value.append(0)
            positive.append(False)
    evaluation['value'] = value
    evaluation['positive'] = positive
    evaluation = evaluation.sort_values(by='value', ascending=False)

    "### Suitable employees"

    for ind, row in evaluation.iterrows():
        with st.container():
            col1, col2 = st.columns([1, 2])
            with col1:
                try:
                    image = Image.open(f"images/{ind}.jpg")
                except FileNotFoundError:
                    image = None
                if image is not None:
                    st.image(image)

            with col2:
                if row['positive']:
                    st.header(f":green[{ind}]")
                else:
                    st.header(f":red[{ind}]")
                st.write(row[0])

