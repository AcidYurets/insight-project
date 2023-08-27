from datetime import timedelta

import pandas as pd
from langchain.llms import Clarifai
from langchain import PromptTemplate, LLMChain
from langchain.chains import SimpleSequentialChain, SequentialChain

CLARIFAI_PAT = 'e5b7191f59214c23ba47f82c50fd6a6f'
clarifai_llm = Clarifai(pat=CLARIFAI_PAT, user_id='meta', app_id='Llama-2', model_id='llama2-7b-chat')

def match_employees(name: str, overview: str, duration: timedelta, goals: str, desired_outcomes: str, sign_of_completion: str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    # Пока pr_sk - такая затычка
    pr_sk = f"{overview}"

    #%%
    template = """Here is a project idea and required skillset: {pr_sk}. Your task is to break the project down into individual steps,
     describe tasks and skills required. output the information in bulletpoints."""

    # template = """Here is a project idea: {project_description} and required skillset {skillset}. Your task is to break the project down into individual steps,
    #  describe tasks and skills required. output the information in bulletpoints."""

    # prompt = PromptTemplate(template=template, input_variables=["project_description", "skillset"])
    prompt = PromptTemplate(template=template, input_variables=["pr_sk"])

    llm_chain_1 = LLMChain(prompt=prompt, llm=clarifai_llm)

    template_2 = """Project plan: [start]{agent_1_output}[/ENDING]. TASK: 
    Based on the given project plan, 
    describe what team members are required to complete the project steps. 
    Provide a list of roles and the corresponding skill set for each role"""

    prompt_2 = PromptTemplate(template=template_2, input_variables=["agent_1_output"])

    input_variables = ["pr_sk", 'agent_1_output']
    # output_variables = ["agent_1_output", "agent_2_output"]

    #llm_chain_1.run(project_description=project_description, skillset=skillset)

    llm_chain_2 = LLMChain(prompt=prompt_2, llm=clarifai_llm)

    chains_1_2 = SimpleSequentialChain(chains=[llm_chain_1, llm_chain_2],
                                       verbose=True,
                                       input_variables=input_variables)

    chains_1_2.run(pr_sk=pr_sk)

    #%%
    #AGENTS 1&2
    pr_sk = """
    PROJECT IDEA: Build a nice house for a young couple.
    REQUIRED SKILLSET: Knowledge of construction and architecture.
    """

    template = """Here is a project idea: {project_description} and required skillset: {skillset}. Your task is to break the project down into individual steps,
     describe tasks and skills required for each step. output the information in bulletpoints."""

    prompt = PromptTemplate(template=template, input_variables=["project_description", 'skillset'])
    llm_chain_1 = LLMChain(prompt=prompt, llm=clarifai_llm)

    template_2 = """Project plan: [start]{agent_1_output}[/ENDING]. TASK: 
    Based on the given project plan, 
    describe what team members are required to complete the project steps. 
    Provide a list of roles and the corresponding skill set for each role"""
    prompt_2 = PromptTemplate(template=template_2, input_variables=["agent_1_output"])

    llm_chain_2 = LLMChain(prompt=prompt_2, llm=clarifai_llm)
    inputs = {"project_description": overview, "skillset": ""}
    output_1 = llm_chain_1.run(inputs)

    output_2 = llm_chain_2.run(output_1)

    print("1&2 complete")
    # input_variables = ["pr_sk", "agent_1_output"]
    # chains_1_2 = SequentialChain(input_variables=input_variables,
    #                              chains=[llm_chain_1, llm_chain_2],
    #                              verbose=True)
    #
    # chains_1_2.run(pr_sk=pr_sk)
    #%%
    #AGENT 3
    employees = pd.read_excel('data_hackaton.xlsx')

    template_3 = """Write a short summary about employee main capabilities, here is employee 
    DESCRIPTION: {description}, SKILLS: {skills}"""
    prompt_3 = PromptTemplate(template=template_3, input_variables=["description", "skills"])
    llm_chain_3 = LLMChain(prompt=prompt_3, llm=clarifai_llm)

    emp_summary = {}

    for i in range(employees.shape[0]):
        name, description, skills = employees.iloc[i,0],employees.iloc[i,1],employees.iloc[i,2]
        inputs = {"description": description, "skills": skills}
        emp_summary[name] = llm_chain_3.run(inputs)

    print("3 complete")
    # llm_chain_3 = LLMChain(prompt=prompt_3, llm=clarifai_llm)
    # inputs = {"name": project_description, "skillset": skillset}
    # output_1 = llm_chain_1.run(inputs)
    #%%
    #AGENT 4
    template_4 = """Based on project plan: {output_1} and required roles: {output_2} provide reasoning why employee should
     or should not participate in the project, here is employee's skillset: {output_3}, be critical."""
    prompt_4 = PromptTemplate(template=template_4, input_variables=["output_1","output_2","output_3"])
    llm_chain_4 = LLMChain(prompt=prompt_4, llm=clarifai_llm)
    emp_reasoning = {}
    for key in emp_summary:
        output_3 = emp_summary[key]
        inputs = {"output_1": output_1, "output_2": output_2, "output_3":output_3}
        emp_reasoning[key] = llm_chain_4.run(inputs)
    print("4 complete")

    #%%
    #AGENT 5
    template_5 = """Based on project requirements and roles: {output_2} and reasoning about employee's participation in the project: {output_4},
    evaluate if it should participate in the project"""
    prompt_5 = PromptTemplate(template=template_5, input_variables=["output_2","output_4"])
    llm_chain_5 = LLMChain(prompt=prompt_5, llm=clarifai_llm)
    emp_evaluation = {}
    for key in emp_reasoning:
        output_4 = emp_summary[key]
        inputs = {"output_2": output_2, "output_4":output_4}
        emp_evaluation[key] = llm_chain_5.run(inputs)
    print("5 complete")

    #%%
    summary_pd = pd.DataFrame([emp_summary]).transpose()
    reasoning_pd = pd.DataFrame([emp_reasoning]).transpose()
    evaluation_pd = pd.DataFrame([emp_evaluation]).transpose()

    return summary_pd, reasoning_pd, evaluation_pd
