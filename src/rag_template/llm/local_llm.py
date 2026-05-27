"""
src/llm/local_llm.py
====================

本地大模型推理模块。

职责：
1. 从本地路径加载 tokenizer 和 causal LM
2. 接收 RAG prompt
3. 调用 model.generate()
4. 返回生成答案

当前主要适配 Qwen2.5-Instruct 这类 decoder-only instruct 模型。
"""

from pathlib import Path
from typing import Optional, Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LocalLLMGenerator:
    """
    本地 LLM 生成器。

    对外只暴露 generate(prompt)。
    RAG 其他模块不关心 tokenizer / model.generate 的细节。
    """

    def __init__(
        self,
        model_name: Union[str, Path],
        device: Optional[str] = None,
    ):
        self.model_name = str(model_name)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        print("=" * 80)
        print("[LocalLLM] 正在加载本地大模型")
        print(f"[LocalLLM] model_name: {self.model_name}")
        print(f"[LocalLLM] device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

        self.model.to(self.device)
        self.model.eval()

        print("[LocalLLM] 模型加载完成")
        print("=" * 80)

    def _build_chat_prompt(self, prompt: str) -> str:
        """
        使用 tokenizer 的 chat_template 构造 Instruct 模型输入。

        Qwen2.5-Instruct 推荐使用 chat template。
        如果 tokenizer 没有 chat_template，则退化为原始 prompt。
        """
        messages = [
            {
                "role": "system",
                "content": "你是一个严谨的知识库问答助手。请严格根据提供的资料回答问题。",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        return prompt

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = False,
    ) -> str:
        """
        根据 prompt 生成回答。

        Args:
            prompt: RAG prompt
            max_new_tokens: 最大新生成 token 数
            temperature: 采样温度
            top_p: nucleus sampling 参数
            do_sample: 是否采样。False 时偏确定性输出。

        Returns:
            answer: 模型生成的答案
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt 不能为空")

        final_prompt = self._build_chat_prompt(prompt)

        encoded = self.tokenizer(
            final_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # 只截取新生成部分，避免把 prompt 一起输出
        new_tokens = output_ids[0][input_ids.shape[-1]:]

        answer = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        )

        return answer.strip()