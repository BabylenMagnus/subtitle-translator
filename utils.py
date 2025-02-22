import os
import re
import yaml
from langchain.prompts import PromptTemplate
from typing import Dict, Optional, Tuple
from langchain_core.messages import AIMessage
from config import PROMPTS_DIR


def load_prompt(prompt_name: str, prompts_dir: str = PROMPTS_DIR) -> PromptTemplate:
    yaml_path = os.path.join(prompts_dir, f"{prompt_name}.yaml")

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Prompt file not found: {yaml_path}")

    with open(yaml_path, 'r', encoding='utf-8') as f:
        prompt_data = yaml.safe_load(f)

    if prompt_name not in prompt_data:
        raise KeyError(f"Prompt '{prompt_name}' not found in {yaml_path}")

    prompt_config = prompt_data[prompt_name]

    prompt = PromptTemplate(
        template=prompt_config['template'], input_variables=prompt_config['input_variables']
    )

    return prompt


def extract_think_and_response(response: str) -> Tuple[Optional[str], str]:
    if isinstance(response, AIMessage):
        response = response.content

    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)

    if think_match:
        think_block = think_match.group(1).strip()
        final_response = response.replace(think_match.group(0), '').strip()
        return think_block, final_response
    elif '<think>' in response:
        final_response = response.replace('<think>', '').strip()
        return None, final_response
    else:
        return None, response
