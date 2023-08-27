import datetime
import os

import streamlit as st

from script import match_employees

"""
# MatchEmp

This app matches employees to the project.
"""

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
desired_outcomes = st.text_area('Input project desired outcomes', placeholder="""The successful completion of this project will result in a cutting-edge recommendation engine integrated into the online streaming platform. The engine will deliver accurate and personalized content suggestions to users, ultimately enhancing their viewing experience, increasing engagement, and contributing to the platform's business success.
""")
sign_of_completion = st.text_area('Input project sign of completion', placeholder="""The project will be considered successful when the recommendation engine demonstrates its ability to provide relevant and engaging content suggestions, positively impacting user engagement metrics. The final deliverables should include comprehensive documentation, model code, integration guidelines, and insights gained from A/B testing.
""")

# Рассчитываем время на выполнение проекта
duration = (complete_by - datetime.date.today())


if st.button('Match employees to this project'):
    PAT = os.environ.get('CLARIFAI_PAT')

    if not PAT:  # If PAT is not set via environment variable
        try:
            PAT = st.secrets['CLARIFAI_PAT']
        except KeyError:
            st.error("Failed to retrieve the Clarifai Personal Access Token!")
            PAT = None

    summary, reasoning, evaluation = match_employees(PAT, name, overview, duration, goals, desired_outcomes, sign_of_completion)
    # st.write(summary)
    # st.write(reasoning)
    # st.write(evaluation)
    print(summary)

