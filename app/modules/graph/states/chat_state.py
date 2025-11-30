from typing import TypedDict, Literal

class ChatState(TypedDict):
    question: str
    category: Literal["info", "lapor", "ticket"]
    answer: str