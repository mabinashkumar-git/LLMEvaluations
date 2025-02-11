import os

import pytest
import requests
from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextPrecisionWithoutReference


@pytest.mark.asyncio
async def test_contextPrecision():
    # create object of a class for that specific metric
    os.environ[
        "OPENAI_API_KEY"] = ""
    llm = ChatOpenAI(model="gpt-4", temperature=0)   #llm object is created
    langChain_llm = LangchainLLMWrapper(llm)         #llm object is converted to RAGAS standard
    contextPrecision = LLMContextPrecisionWithoutReference(llm=langChain_llm)
    question = "How many articles are there in the Selenium webdriver python course?"

    # Feed data
    responseDict = requests.post("https://rahulshettyacademy.com/rag-llm/ask",
                                 json={
                                     "question": question,
                                     "chat_history": [
                                     ]
                                 }).json()

    print(responseDict)

    sample = SingleTurnSample(
        user_input=question,
        response=responseDict["answer"],
        retrieved_contexts=[responseDict["retrieved_docs"][0]["page_content"],
                            responseDict["retrieved_docs"][1]["page_content"],
                            responseDict["retrieved_docs"][2]["page_content"]]
    )

    #score
    score = await contextPrecision.single_turn_ascore(sample)
    print(score)
    assert score > 0.8

    # sample = SingleTurnSample(
    #     user_input="How many articles are there in the Selenium webdriver python course?",
    #     response="There are 23 articles in the Selenium WebDriver Python course.  \n",
    #     retrieved_contexts=[
    #         "Complete Understanding on Selenium Python API Methods with real time Scenarios on LIVE Websites\n\""
    #         "Last but not least\" you can clear any Interview and can Lead Entire Selenium Python Projects from Design "
    #         "Stage\nThis course includes:\n17.5 hours on-demand video\nAssignments\n23 articles\n9 downloadable resources"
    #         "\nAccess on mobile and TV\nCertificate of completion\nRequirements",
    #         "What you'll learn\n*****By the end of this course,You will be Mastered on Selenium Webdriver with strong Core "
    #         "JAVA basics\n****You will gain the ability to design PAGEOBJECT, DATADRIVEN&HYBRID Automation FRAMEWORKS from "
    #         "scratch\n*** InDepth understanding of real time Selenium CHALLENGES with 100 + examples\n*Complete knowledge on TestNG, "
    #         "MAVEN,ANT, JENKINS,LOG4J, CUCUMBER, HTML REPORTS,EXCEL API, GRID PARALLEL TESTING"]
    # )
