#!/usr/bin/env python3
"""Browser UI for script chunking."""

from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr

from script_chunker import clean_text, split_into_scenes, split_into_scenes_contextual


def chunk_from_text(raw_text: str, words_per_scene: int, mode: str):
    if not raw_text or not raw_text.strip():
        return "Please provide script text.", ""

    cleaned_text = clean_text(raw_text)
    if mode == "Context-Aware":
        scenes = split_into_scenes_contextual(cleaned_text, words_per_scene)
    else:
        scenes = split_into_scenes(cleaned_text, words_per_scene)
    output_text = "\n".join(scenes)

    total_words = len(cleaned_text.split())
    total_scenes = len(scenes)
    runtime_minutes = (total_scenes * 5) / 60

    stats = (
        f"Total words: {total_words}\n"
        f"Total scenes: {total_scenes}\n"
        f"Mode: {mode}\n"
        f"Words per scene: ~{(total_words / total_scenes):.1f}\n"
        f"Estimated runtime: {runtime_minutes:.1f} minutes"
    )
    return output_text, stats


def chunk_from_file(uploaded_file, words_per_scene: int, mode: str):
    if uploaded_file is None:
        return "Please upload a .txt file.", ""

    file_path = Path(uploaded_file)
    raw_text = file_path.read_text(encoding="utf-8")
    return chunk_from_text(raw_text, words_per_scene, mode)


with gr.Blocks(title="Script Chunker") as demo:
    gr.Markdown("# Script Chunker")
    gr.Markdown("Split scripts into `[SCENE XXX]` chunks for storyboard generation.")

    with gr.Row():
        words_per_scene = gr.Slider(
            minimum=6,
            maximum=20,
            value=11,
            step=1,
            label="Words per scene",
        )
        mode = gr.Radio(
            choices=["Basic", "Context-Aware"],
            value="Context-Aware",
            label="Split mode",
        )

    with gr.Tab("Paste Text"):
        text_input = gr.Textbox(
            label="Script text",
            lines=14,
            placeholder="Paste your script here...",
        )
        text_run = gr.Button("Chunk Script")

    with gr.Tab("Upload File"):
        file_input = gr.File(
            label="Upload .txt script",
            file_types=[".txt"],
            type="filepath",
        )
        file_run = gr.Button("Chunk File")

    output = gr.Textbox(label="Chunked output", lines=16)
    copy_btn = gr.Button("Copy Output")
    stats = gr.Textbox(label="Stats", lines=4)
    download = gr.File(label="Download output")

    def write_output_file(output_text: str):
        if not output_text or output_text.startswith("Please "):
            return None
        tmp = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".txt", delete=False
        )
        tmp.write(output_text)
        tmp.flush()
        tmp.close()
        return tmp.name

    text_run.click(
        fn=chunk_from_text,
        inputs=[text_input, words_per_scene, mode],
        outputs=[output, stats],
    ).then(
        fn=write_output_file,
        inputs=[output],
        outputs=[download],
    )

    file_run.click(
        fn=chunk_from_file,
        inputs=[file_input, words_per_scene, mode],
        outputs=[output, stats],
    ).then(
        fn=write_output_file,
        inputs=[output],
        outputs=[download],
    )

    copy_btn.click(
        fn=None,
        inputs=[output],
        outputs=[],
        js="(text) => { if (text) { navigator.clipboard.writeText(text); } }",
    )


if __name__ == "__main__":
    demo.launch()
