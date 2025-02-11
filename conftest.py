import os

import pytest
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

# This is a global file which all test can access

os.environ[
    "OPENAI_API_KEY"] = "provide open api key here"

@pytest.fixture
def llmWrapper():
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    lang_chain_llm = LangchainLLMWrapper(llm)
    return lang_chain_llm
