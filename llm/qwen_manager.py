from functools import lru_cache
from typing import Any

import torch
from langchain_core.language_models.llms import LLM
from pydantic import ConfigDict
from transformers import AutoModelForCausalLM, AutoTokenizer

from config.settings import HF_TOKEN, MAX_NEW_TOKENS, RAG_ANSWER_MODEL_NAME


class QwenTransformersLLM(LLM):
    tokenizer: Any
    model: Any
    model_name: str = RAG_ANSWER_MODEL_NAME
    max_new_tokens: int = MAX_NEW_TOKENS
    temperature: float = 0.1
    top_p: float = 0.9

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def _llm_type(self) -> str:
        return "qwen_transformers"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

    def _call(self, prompt: str, stop: list[str] | None = None, run_manager: Any = None, **kwargs: Any) -> str:
        rendered_prompt = self._build_chat_prompt(prompt)
        model_inputs = self.tokenizer(rendered_prompt, return_tensors="pt")
        model_inputs = {key: value.to(self._resolve_device()) for key, value in model_inputs.items()}

        generation_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.max_new_tokens),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)
        if temperature and temperature > 0:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": temperature,
                    "top_p": top_p,
                }
            )
        else:
            generation_kwargs["do_sample"] = False

        with torch.no_grad():
            generated_ids = self.model.generate(**model_inputs, **generation_kwargs)

        generated_tokens = generated_ids[0][model_inputs["input_ids"].shape[-1] :]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        if stop:
            response = self._apply_stop_tokens(response, stop)
        return response

    def _build_chat_prompt(self, prompt: str) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def _resolve_device(self) -> str | int:
        device_map = getattr(self.model, "hf_device_map", None)
        if device_map:
            for device in device_map.values():
                if device != "disk":
                    return device
        return str(self.model.device)

    @staticmethod
    def _apply_stop_tokens(text: str, stop: list[str]) -> str:
        trimmed_text = text
        for token in stop:
            if token in trimmed_text:
                trimmed_text = trimmed_text.split(token)[0]
        return trimmed_text.strip()


@lru_cache(maxsize=4)
def load_qwen_llm(
    model_name: str = RAG_ANSWER_MODEL_NAME,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = 0.1,
    top_p: float = 0.9,
) -> QwenTransformersLLM:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=HF_TOKEN)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=HF_TOKEN,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    return QwenTransformersLLM(
        tokenizer=tokenizer,
        model=model,
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
