import pytest
from ragas import SingleTurnSample
from ragas.metrics import Faithfulness

from conftest import llmWrapper
from utils import load_test_data, get_llm_response


@pytest.mark.parametrize("getData", load_test_data("Test4.json"), indirect=True)
@pytest.mark.asyncio
async def test_faithfulness(llmWrapper, getData):
    faithful = Faithfulness(llm=llmWrapper)
    score = await faithful.single_turn_ascore(getData)
    print(score)
    assert score > 0.8


@pytest.fixture
# def __getattr__(name):
def getData(request):
    test_data = request.param
    responseDict = get_llm_response(test_data)

    sample = SingleTurnSample(
        user_input=test_data["question"],
        response=responseDict["answer"],
        # retrieved_contexts=[responseDict["retrieved_docs"][0]["page_content"]]
        retrieved_contexts=[doc["page_content"] for doc in responseDict.get("retrieved_docs")]
    )
    return sample
