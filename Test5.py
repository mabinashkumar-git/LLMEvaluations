import os

# import pytest
# from langsmith import evaluate
# from ragas import SingleTurnSample, EvaluationDataset
# from ragas.metrics import ResponseRelevancy, FactualCorrectness
#
# from utils import get_llm_response, load_test_data

import pytest
from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.metrics import ResponseRelevancy, FactualCorrectness

from utils import load_test_data, get_llm_response

#Get your token from https://app.ragas.io/dashboard -> Click on App token -> Then click Create new token
os.environ["RAGAS_APP_TOKEN"] = "apt.4526-7a8a3c34c95e-ec88-b132-1553d88c-66959"

@pytest.mark.parametrize("getData", load_test_data("Test5.json"), indirect=True)
@pytest.mark.asyncio
async def test_relevancy_factual(llmWrapper, getData):
    metrics = [ResponseRelevancy(llm=llmWrapper),
               FactualCorrectness(llm=llmWrapper)]

    eval_dataset = EvaluationDataset([getData])     #Converting SingleTurnSample in to Raga understandable data set
    results = evaluate(datasets=eval_dataset, metrics=metrics)
    # results = evaluate(datasets=eval_dataset)
    print(results)
    print(results["answer_relevancy"])
    results.upload()                     # Results will be uploaded to cloud dashboard https://app.ragas.io/dashboard


@pytest.fixture
def getData(request):
    test_data = request.param
    responseDict = get_llm_response(test_data)

    sample = SingleTurnSample(
        user_input=test_data["question"],
        response=responseDict["answer"],
        # retrieved_contexts=[responseDict["retrieved_docs"][0]["page_content"]]
        retrieved_contexts=[doc["page_content"] for doc in responseDict.get("retrieved_docs")],
        reference=test_data["reference"]
    )
    return sample
