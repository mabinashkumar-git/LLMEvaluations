import os

import pytest
import requests
from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall
from sqlalchemy import True_

from utils import get_llm_response, load_test_data

@pytest.mark.asyncio
@pytest.mark.parametrize("getdata",
                         load_test_data("Test3.json"),
                         indirect=True                    #LC:25
                         )

async def test_context_recall(llmWrapper, getdata):

    context_recall = LLMContextRecall(llm=llmWrapper)
    score = await context_recall.single_turn_ascore(getdata)
    print(score)
    assert score > 0.7


@pytest.fixture
def getdata(request):
    test_data = request.param
    print(test_data)
    responseDict = get_llm_response(test_data)

    sample = SingleTurnSample(
        user_input=test_data["question"],
        retrieved_contexts=[responseDict["retrieved_docs"][0]["page_content"],
                            responseDict["retrieved_docs"][1]["page_content"],
                            responseDict["retrieved_docs"][2]["page_content"]],
        reference=test_data["reference"]
    )
    return sample
