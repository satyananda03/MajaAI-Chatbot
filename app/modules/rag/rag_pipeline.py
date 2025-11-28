import os 
from app.core.config import settings
from operator import itemgetter
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from app.modules.rag.retriever import ParentChildRetriever
from app.services.llm import get_llm
from app.services.prompt import prompt_template
from app.utils.parser import document_parser, url_parser

class RAGPipeline:
    def __init__(self, prompt_version: str):
        self.retriever = ParentChildRetriever()
        self.llm = get_llm()
        self.prompt = prompt_template.get_prompt(prompt_version)
        self.output_parser = StrOutputParser()

    def chat(self):
        retrieval = RunnableParallel({
                                    "docs": itemgetter("question") | self.retriever, 
                                    "question": itemgetter("question")
                                    })
        chat_pipeline = (
                        retrieval
                        | RunnableParallel({    
                                            "context": lambda x: document_parser(x["docs"]),
                                            "url": lambda x: url_parser(x["docs"]),
                                            "question": itemgetter("question")
                                            })
                        | self.prompt 
                        | self.llm
                        | self.output_parser
                        )
        return chat_pipeline

    def evaluate(self):
        retrieval = RunnableParallel({
            "docs": itemgetter("question") | self.retriever, 
            "question": itemgetter("question")
        })
        eval_pipeline = (
            retrieval
            | RunnableParallel({
                "context": lambda x: document_parser(x["docs"]),
                "url": lambda x: url_parser(x["docs"]),
                "question": itemgetter("question") 
            })
            | RunnableParallel({
                "answer": self.prompt | self.llm | self.output_parser,
                "context": itemgetter("context")
            })
        )
        return eval_pipeline

    def evaluate_retriever(self):
        eval_pipeline = RunnablePassthrough.assign(context = itemgetter("question") | self.retriever | document_parser)
        return eval_pipeline