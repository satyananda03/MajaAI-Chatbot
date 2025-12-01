from operator import itemgetter
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from app.modules.rag.retriever import ParentChildRetriever
from app.services.llm import get_llm
from app.services.prompt import prompt_template
from app.utils.parser import context_parser, url_parser

class RAGPipeline:
    def __init__(self):
        self.retriever = ParentChildRetriever()
        self.llm = get_llm(max_tokens=1000)
        self.prompt = prompt_template.get_prompt("rag")
        self.output_parser = StrOutputParser()

    def chat(self):
        chat_pipeline = (
            RunnablePassthrough.assign(docs=itemgetter("question") | self.retriever)
            | RunnablePassthrough.assign(
                                        context=lambda x: context_parser(x["docs"]),
                                        url=lambda x: url_parser(x["docs"])
                                        )
            | self.prompt 
            | self.llm
            | self.output_parser
        )
        return chat_pipeline
        
    def evaluate(self):
        eval_pipeline = (
            RunnablePassthrough.assign(docs=itemgetter("question") | self.retriever)
            | RunnablePassthrough.assign(
                                        context=lambda x: context_parser(x["docs"]),
                                        url=lambda x: url_parser(x["docs"])
                                        )
            | RunnablePassthrough.assign(answer=self.prompt | self.llm | self.output_parser)
        )
        return eval_pipeline

    def evaluate_retriever(self):
        eval_pipeline = RunnablePassthrough.assign(context = itemgetter("question") 
                        | self.retriever 
                        | (lambda docs: [context_parser(docs)]))
        return eval_pipeline