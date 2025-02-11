import pytest
from ragas import SingleTurnSample, MultiTurnSample
from ragas.messages import HumanMessage, AIMessage
from ragas.metrics import Faithfulness, TopicAdherenceScore

from conftest import llmWrapper
from utils import load_test_data, get_llm_response


# @pytest.mark.parametrize("getData", load_test_data("Test4.json"), indirect=True)
@pytest.mark.asyncio
async def test_topicAdherence(llmWrapper, getData):
    topicScore = TopicAdherenceScore(llm=llmWrapper)       #TopicAdherenceScore i for multi conversion
    score = await topicScore.multi_turn_ascore(getData)
    print(score)
    assert score > 0.8


@pytest.fixture
# def __getattr__(name):
def getData(request):
    # test_data = request.param
    # responseDict = get_llm_response(test_data)
    conversation = [
        HumanMessage(content="How many articles are there in the Selenium webdriver python course?"),
        AIMessage(content="There are 23 articles in the Selenium Webdriver Python course"),
        HumanMessage(content="How many downloadable resources are there in this course?"),
        AIMessage(content="There are 9 downloadable resources in the course.")
    ]
    references = [""" 
    The AI should:
    1. Give results related to selenium webdriver python course
    2. There 23 articles and 9 downloadable resources in the course"""]

    sample = MultiTurnSample(
        user_input=conversation,
        reference_topics=references
    )
    return sample
