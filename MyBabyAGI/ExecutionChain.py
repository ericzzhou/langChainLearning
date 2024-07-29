# 导入所需的库和模块
from collections import deque
from typing import Dict, List, Optional, Any
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Field
from langchain.chains.base import Chain
from langchain_community.vectorstores import FAISS
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore


# 任务执行链
class ExecutionChain(LLMChain):
    """负责执行任务的链"""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """从LLM获取响应解析器"""
        execution_template = (
            "你是一个基于以下目标执行一项任务的AI助手: {objective}."
            " 考虑这些以前完成的任务: {context}."
            " 你的任务: {task}."
            " Response:"
        )
        prompt = PromptTemplate(
            template=execution_template,
            input_variables=["objective", "context", "task"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)
