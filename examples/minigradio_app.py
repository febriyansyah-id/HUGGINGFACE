#!/usr/bin/env python
"""
Spin-up a lightweight Gradio interface backed by whatever model you supply –
perfect companion to `make deploy` afterwards.

Demo:
    python examples/minigradio_app.py --model sshleifer/distilbart-cnn-12-6
Then visit http://localhost:7860
"""

from __future__ import annotations

import argparse

SUMMARISE_HELP = "Summarise the article."

EXAMPLE_PARAGRAPH = (
    "Large-scale neural networks have revolutionised NLP, enabling machines to reason "
    "about unstructured text almost indistinguishably from humans. Yet deploying such "
    "systems responsibly remains challenging due to cost, latency and bias considerations. "
    "Researchers increasingly rely on distillation techniques to compress heavyweight "
    "teachers into agile student models suitable for edge devices."
)

parser = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--model", default="sshleifer/distilbart-cnn-12-6")
parser.add_argument("--share-link", action="store_true", help="Expose publicly via tunnel.")
parser.add_argument("--server-port", type=int, default=7860)


def summarise_long_form(raw_body: str) -> str:
    payload = SUMMARISER(raw_body, max_length=130, min_length=34, do_sample=False)[0]["summary_text"]
    return payload


if __name__ == "__main__":
    opts = parser.parse_args()

    import gradio as gr
    from transformers import pipeline

    SUMMARISER = pipeline("summarization", model=opts.model, device_map="auto")

    ui = gr.Interface(
        fn=summarise_long_form,
        title="Long-form Summariser (\ud83e\udd16 powered by HF)",
        description=SUMMARISE_HELP,
        theme=gr.themes.Soft(primary_hue="teal"),
        inputs=[
            gr.TextArea(label="Article body", placeholder="Paste anything …", lines=8),
        ],
        outputs=gr.Markdown(label="Condensed digest"),
        examples=[[EXAMPLE_PARAGRAPH]],
        allow_flagging="never",
    )

    ui.queue(default_concurrency_limit=4).launch(
        server_port=opts.server_port,
        share=bool(opts.share_link),
        prevent_thread_lock=True,
    )
