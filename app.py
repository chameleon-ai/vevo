import errno
import os
import sys
import torch
import traceback
import tkinter as tk
from tkinter import filedialog
sys.path.append('./Amphion') # For importing modules relative to the Amphion directory
import Amphion.models.vc.vevo.vevo_utils as vevo_utils
from huggingface_hub import snapshot_download

def infer():
    try:
        content_filename = content_path.get()
        reference_filename = reference_path.get()
        output_dir = output_path.get()
        if not os.path.isdir(output_dir):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), output_dir)
        if not os.path.isfile(content_filename):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), content_filename)
        if not os.path.isfile(reference_filename):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), reference_filename)
        
        output_filename = os.path.join(output_dir, 'output.wav')
        
        gen_audio = inference_pipeline.inference_ar_and_fm(
            src_wav_path=content_filename,
            src_text=None,
            style_ref_wav_path=reference_filename,
            timbre_ref_wav_path=reference_filename,
        )
        vevo_utils.save_audio(gen_audio, output_path=output_filename)
        message = "Done. Output file: '{}'".format(output_filename)
        print(message)
        error_str.set("")
    except Exception as e:
        error_str.set(str(e))
        traceback.print_exc()

def load_model():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Content Tokenizer
    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir="./ckpts/Vevo",
        allow_patterns=["tokenizer/vq32/*"],
    )
    content_tokenizer_ckpt_path = os.path.join(
        local_dir, "tokenizer/vq32/hubert_large_l18_c32.pkl"
    )

    # Content-Style Tokenizer
    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir="./ckpts/Vevo",
        allow_patterns=["tokenizer/vq8192/*"],
    )
    content_style_tokenizer_ckpt_path = os.path.join(local_dir, "tokenizer/vq8192")

    # Autoregressive Transformer
    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir="./ckpts/Vevo",
        allow_patterns=["contentstyle_modeling/Vq32ToVq8192/*"],
    )
    ar_cfg_path = "./config/Vq32ToVq8192.json"
    ar_ckpt_path = os.path.join(local_dir, "contentstyle_modeling/Vq32ToVq8192")

    # Flow Matching Transformer
    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir="./ckpts/Vevo",
        allow_patterns=["acoustic_modeling/Vq8192ToMels/*"],
    )
    fmt_cfg_path = "./config/Vq8192ToMels.json"
    fmt_ckpt_path = os.path.join(local_dir, "acoustic_modeling/Vq8192ToMels")

    # Vocoder
    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir="./ckpts/Vevo",
        allow_patterns=["acoustic_modeling/Vocoder/*"],
    )
    vocoder_cfg_path = "./Amphion/models/vc/vevo/config/Vocoder.json"
    vocoder_ckpt_path = os.path.join(local_dir, "acoustic_modeling/Vocoder")

    # Inference
    pipeline = vevo_utils.VevoInferencePipeline(
        content_tokenizer_ckpt_path=content_tokenizer_ckpt_path,
        content_style_tokenizer_ckpt_path=content_style_tokenizer_ckpt_path,
        ar_cfg_path=ar_cfg_path,
        ar_ckpt_path=ar_ckpt_path,
        fmt_cfg_path=fmt_cfg_path,
        fmt_ckpt_path=fmt_ckpt_path,
        vocoder_cfg_path=vocoder_cfg_path,
        vocoder_ckpt_path=vocoder_ckpt_path,
        device=device
    )
    return pipeline

def browse_reference():
    filename = filedialog.askopenfilename(filetypes=(("wav files","*.wav"),("All files","*.*")))
    reference_entry.insert(tk.END, filename)

def browse_content():
    filename = filedialog.askopenfilename(filetypes=(("wav files","*.wav"),("All files","*.*")))
    content_entry.insert(tk.END, filename)

def browse_output():
    filename = filedialog.askdirectory()
    output_entry.insert(tk.END, filename)

if __name__ == '__main__':
    inference_pipeline = load_model()
    root = tk.Tk()
    root.title('Vevo GUI')
    root.geometry("800x600")

    tk.Grid.columnconfigure(root, 1, weight=1) # Weight this column to have it stretch with the window

    reference_label = tk.Label(root, text='Reference voice:')
    reference_path = tk.StringVar()
    reference_path.set('./Amphion/models/vc/vevo/wav/arabic_male.wav')
    reference_entry = tk.Entry(root, textvariable=reference_path)
    reference_browse = tk.Button(root, text='Browse', command=browse_reference)
    content_label = tk.Label(root, text='Content audio:')
    content_path = tk.StringVar()
    content_path.set('./Amphion/models/vc/vevo/wav/source.wav')
    content_entry = tk.Entry(root, textvariable=content_path)
    content_browse = tk.Button(root, text='Browse', command=browse_content)
    output_label = tk.Label(root, text='Output Directory:')
    output_path = tk.StringVar()
    output_path.set('./')
    output_entry = tk.Entry(root, textvariable=output_path)
    output_browse = tk.Button(root, text='Browse', command=browse_output)

    infer_button = tk.Button(root, text='Run Inference', command=infer)

    error_str = tk.StringVar()
    error_label = tk.Label(root, textvariable=error_str)

    
    reference_label.grid(row=0,column=0)
    reference_entry.grid(row=0,column=1, sticky=tk.EW)
    reference_browse.grid(row=0,column=2)
    content_label.grid(row=1,column=0)
    content_entry.grid(row=1,column=1, sticky=tk.EW)
    content_browse.grid(row=1,column=2)
    output_label.grid(row=2,column=0)
    output_entry.grid(row=2,column=1, sticky=tk.EW)
    output_browse.grid(row=2,column=2)
    infer_button.grid(row=3,column=1)
    error_label.grid(row=4,column=1)
    root.mainloop()

