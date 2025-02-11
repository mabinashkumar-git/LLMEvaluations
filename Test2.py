import os

import pytest
import requests
from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall

os.environ[
    "OPENAI_API_KEY"] = "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"


@pytest.mark.asyncio
async def test_context_recall():
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    lang_chain_llm = LangchainLLMWrapper(llm)
    context_recall = LLMContextRecall(llm=lang_chain_llm)
    question = "How many articles are there in the Selenium webdriver python course?"

    responseDict = requests.post("https://rahulshettyacademy.com/rag-llm/ask",
                                 json={
                                     "question": question,
                                     "chat_history": [
                                     ]
                                 }).json()

    sample = SingleTurnSample(
        user_input=question,
        retrieved_contexts=[responseDict["retrieved_docs"][0]["page_content"],
                            responseDict["retrieved_docs"][1]["page_content"],
                            responseDict["retrieved_docs"][2]["page_content"]],
        reference="23"
    )
    score = await context_recall.single_turn_ascore(sample)
    print(score)
    assert score > 0.7
