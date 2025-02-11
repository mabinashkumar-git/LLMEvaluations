import os
from pathlib import Path

import nltk.data
from langchain_community.document_loaders import DirectoryLoader, UnstructuredWordDocumentLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator

os.environ[
    "OPENAI_API_KEY"] = ""
os.environ["RAGAS_APP_TOKEN"] = "apt.4526-7a8a3c34c95e-ec88-b132-1553d88c-66959"


def test_dataCreation():
    nltk.data.path.append("/Users/abinash/Documents/Personal/Code Worspace/LLMEvaluation/nltk_data/")
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    langChain_llm = LangchainLLMWrapper(llm)
    embed = OpenAIEmbeddings()
    # projectDirectory = Path(__file__).parent.absolute()
    # test_data_path = projectDirectory/"Documents"
    loader = DirectoryLoader(
        # path=test_data_path,
        path="/Users/abinash/Documents/Personal/Code Worspace/LLMEvaluation/Documents/",
        glob="**/*.docx",
        loader_cls=UnstructuredWordDocumentLoader
    )
    docs = loader.load()
    generate_embeddings = LangchainEmbeddingsWrapper(embed)
    generator = TestsetGenerator(llm=langChain_llm, embedding_model=generate_embeddings) # embedding_model -> Converts word document into vector format, from vector format to create it needs "llm" intelligence
    dataset = generator.generate_with_langchain_docs(docs, testset_size=20)
    print(dataset)
    dataset.upload()     # Upload to Ragas portal
