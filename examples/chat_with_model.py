#!/usr/bin/env python
"""
Zero-code way to chat with ANY causal LLM hosted on the Hub —
great sanity check before wiring something heavier.

Try:
    python examples/chat_with_model.py --model mistralai/Ministral-8B-Instruct-2410
    python examples/chat_with_model.py --model meta-llama/Llama-3.2-1B-Instruct
"""

from __future__ import annotations

import argparse

PROMPT_ROUTINE = [
    {"role": "system", "content": "Answer briefly and accurately."},
    {"role": "user", "content": "{QUESTION}"},
]

QUESTIONS = [
    "Explain quantum entanglement in one paragraph.",
    "Write a haiku about distributed systems.",
    "What is the capital of Indonesia? Answer concisely.",
]

parser = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--model", default="mistralai/Ministral-8B-Instruct-2410")
parser.add_argument("--temperature", type=float, default=0.65)
parser.add_argument("--top-k", type=int, default=40)
parser.add_argument("--max-new-tokens", type=int, default=220)
parser.add_argument("--repeat-one-q", default=None, help="Ask a single arbitrary question.")


def assemble_conversation(user_query: str):
    convo = PROMPT_ROUTINE.copy()
    convo[1]["content"] = convo[1]["content"].format(QUESTION=user_query)
    return convo


def stream_answer(client_, conv_history):
    reply_iterator = client_.generate(
        conv_history,
        temperature=ARGS.temperature,
        top_k=ARGS.top_k,
        max_new_tokens=ARGS.max_new_tokens,
        repetition_penalty=1.025,
    )
    collected = ""
    for chunk in reply_iterator:
        piece = chunk.outputs[0].text
        collected += piece
        yield piece
    return collected


if __name__ == "__main__":
    ARGS = parser.parse_args()

    from rich.console import Console
    from rich.markdown import Markdown
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    console_printer = Console(width=110)

    processor_instance = AutoProcessor.from_pretrained(ARGS.model)
    llm = Gemma3ForConditionalGeneration.from_pretrained(
        ARGS.model,
        device_map="auto",
        torch_dtype="auto",
    )

    candidates_q = QUESTIONS if not ARGS.repeat_one_q else [ARGS.repeat_one_q]

    for q_no, question in enumerate(candidates_q, start=1):
        console_printer.print(f"\n[b cyan]\u2776 Question #{q_no}[/]", style="bold underline magenta")
        console_printer.print("[dim italic]" + question + "[/]")

        transcript = assemble_conversation(question)
        rendered_block = ""
        console_printer.print("\n[yellow bold]Assistant:[yellow] ")
        for fragment in stream_answer(llm, transcript):
            rendered_block += fragment
        console_printer.print(Markdown(rendered_block))
