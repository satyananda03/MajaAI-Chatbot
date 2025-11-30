import os
import yaml
from langchain_core.prompts import (ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate)
from app.core.config import settings
class Prompt:
    def __init__(self, base_dir=settings.PROMPT_DIR):
        self.base_dir = os.path.join(os.getcwd(), base_dir, settings.PROMPT_VERSION)
        self._cache = {}

    def get_prompt(self, name: str) -> ChatPromptTemplate:
        if name in self._cache:
            return self._cache[name]
        file_path = os.path.join(self.base_dir, f"{name}.yaml")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Prompt not found: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        system_msg = SystemMessagePromptTemplate.from_template(data["system_template"])
        human_msg = HumanMessagePromptTemplate.from_template(data["human_template"])
        chat = ChatPromptTemplate.from_messages([system_msg, human_msg])
        self._cache[name] = chat
        return chat

prompt_template = Prompt()