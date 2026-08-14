import gradio as gr
import spaces
import torch
from omnivoice import OmniVoice
import soundfile as sf, io

MODEL = None

def get_model():
    global MODEL
    if MODEL is None:
        MODEL = OmniVoice.from_pretrained(
            "k2-fsa/OmniVoice",
            device_map="cuda",
            dtype=torch.float16
        )
    return MODEL

@spaces.GPU(duration=120)
def synthesize(text, instruct="", ref_audio=None, ref_text=""):
    model = get_model()
    kwargs = {"text": text}
    if ref_audio: kwargs["ref_audio"] = ref_audio
    if ref_text: kwargs["ref_text"] = ref_text
    if instruct: kwargs["instruct"] = instruct
    audio = model.generate(**kwargs)
    buf = io.BytesIO()
    sf.write(buf, audio[0], 24000, format="wav")
    return buf.getvalue()

with gr.Blocks() as demo:
    gr.Markdown("## OmniVoice")
    text = gr.Textbox(label="Text")
    instruct = gr.Textbox(label="Voice Style (optional: 'female, british accent')")
    ref = gr.Audio(type="filepath", label="Reference Audio (optional)")
    ref_text = gr.Textbox(label="Reference Text (optional)")
    out = gr.Audio(label="Output")
    gr.Button("Generate").click(synthesize, [text, instruct, ref, ref_text], out)

demo.launch()
