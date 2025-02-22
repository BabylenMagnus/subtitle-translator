from typing import Optional, List, Dict
from langchain_ollama import OllamaLLM
from langchain_groq import ChatGroq
from langchain_core.rate_limiters import InMemoryRateLimiter
import time
from utils import load_prompt


class BasePromptService:
    def __init__(self, temperature: float = 0.5, model: str = "deepseek-r1:32b",
            prompt_name: str = "subtitle_translator", local: bool = True):
        if local:
            self.llm = OllamaLLM(
                model=model, options={"temperature": temperature}
            )
        else:  # cloud API (Groq)
            rate_limiter = InMemoryRateLimiter(
                requests_per_second=0.5, max_bucket_size=30, check_every_n_seconds=0.5
            )

            self.llm = ChatGroq(
                model=model, temperature=temperature, rate_limiter=rate_limiter
            )

        self.prompt = load_prompt(prompt_name)

    def _invoke_llm(self, formatted_prompt: str) -> str:
        try:
            response = self.llm.invoke(formatted_prompt)
            return response
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                print("Rate limit hit, retrying after delay...")
                time.sleep(2)
                return self._invoke_llm(formatted_prompt)
            raise


class SubtitleTranslator(BasePromptService):
    def __init__(self, temperature: float = 0.3, model: str = "deepseek-r1:32b", local: bool = True,
            context_lines: int = 10):
        super().__init__(
            temperature=temperature, model=model, local=local
        )
        self.context_lines = context_lines

    def _get_context(self, subtitles: List[Dict], current_idx: int) -> str:
        """Get previous lines as context"""
        start_idx = max(0, current_idx - self.context_lines)
        context_subs = subtitles[start_idx:current_idx]
        return "\n".join(sub["text"] for sub in context_subs)

    def translate_subtitles(self, subtitles: List[Dict], target_lang: str) -> List[Dict]:
        """
        Translate subtitle entries while maintaining timing and structure.
        
        Args:
            subtitles: List of subtitle dictionaries with index, timestamp, text
            target_lang: Target language for translation
            
        Returns:
            List of translated subtitle dictionaries
        """
        translated_subs = []

        for i, subtitle in enumerate(subtitles):
            context = self._get_context(subtitles, i)

            formatted_prompt = self.prompt.format(
                context=context, current_line=subtitle["text"]
            )

            translated_text = self._invoke_llm(formatted_prompt)

            translated_subs.append(
                {"index": subtitle["index"], "timestamp": subtitle["timestamp"], "text": translated_text.strip()}
            )

            if i % 10 == 0:  # Progress indicator
                print(f"Translated {i}/{len(subtitles)} lines")

        return translated_subs
