import os
import yaml
from langchain_core.prompts import (ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate)
from app.core.config import settings
class Prompt:
    def __init__(self, template_dir: str = settings.PROMPT_DIR):
        self.template_dir = os.path.join(os.getcwd(), template_dir)
        self._prompt_cache = {}
    def get_prompt(self, version: str) -> ChatPromptTemplate:
        if version in self._prompt_cache:
            return self._prompt_cache[version]
        # Load file prompt template
        file_path = os.path.join(self.template_dir, f"{version}.yaml")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Error parsing YAML prompt {e}")
        # LangChain Prompt Template
        system_message = SystemMessagePromptTemplate.from_template(data['system_template'])
        human_message = HumanMessagePromptTemplate.from_template(data['human_template'])
        chat_template = ChatPromptTemplate.from_messages([system_message, human_message])
        self._prompt_cache[version] = chat_template
        return chat_template

prompt_template = Prompt()