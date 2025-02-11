import os

import pytest
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

# This is a global file which all test can access

os.environ[
    "OPENAI_API_KEY"] = "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"

@pytest.fixture
def llmWrapper():
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    lang_chain_llm = LangchainLLMWrapper(llm)
    return lang_chain_llm
