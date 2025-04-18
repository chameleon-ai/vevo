import errno
import mimetypes
import os
import sys
import traceback
from pydub import AudioSegment
import torch
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk
sys.path.append('./Amphion') # For importing modules relative to the Amphion directory
import Amphion.models.svc.vevosing.vevosing_utils as vevosing_utils
from huggingface_hub import snapshot_download

def get_unique_filename(basename : str, extension : str):
    filename = '{}.{}'.format(basename,extension)
    x = 0
    while os.path.isfile(filename):
        x += 1
        filename = '{}-{}.{}'.format(basename,x,extension)
    return filename

# Converts an audio file to wav if needed
def get_wav(filename, out_dir):
    # Possible mime types: https://www.iana.org/assignments/media-types/media-types.xhtml
    mime, encoding = mimetypes.guess_type(filename)
    if mime == 'audio/wav' or mime == 'audio/x-wav':
        return filename
    elif mime == 'audio/mpeg':
        seg = AudioSegment.from_mp3(filename)
        # Create a new file in the output directory named after the input
        wav_filename = get_unique_filename(os.path.join(out_dir, os.path.splitext(os.path.basename(filename))[0]), 'wav')
        print(wav_filename)
        seg.export(wav_filename, format="wav")
        return wav_filename
    else:
        raise RuntimeError("Unsupported file type {} for file '{}'".format(mime, filename))

# Do vevo inference based on the provided mode string
def run_inference(pipeline : vevosing_utils.VevosingInferencePipeline,
                  mode : str,
                  content : str,
                  ref_style : str,
                  ref_timbre : str,
                  src_text : str,
                  ref_text : str,
                  src_language : str,
                  ref_language : str,
                  steps : int):
    if mode == 'style':
        return pipeline.inference_ar_and_fm(
            task="recognition-synthesis",
            src_wav_path=content,
            src_text=src_text,
            style_ref_wav_path=ref_style,
            style_ref_wav_text=ref_text,
            src_text_language=src_language,
            style_ref_wav_text_language=ref_language,
            timbre_ref_wav_path=ref_timbre,
            use_style_tokens_as_ar_input=True,  # To use the prosody code of the raw wav
            flow_matching_steps=steps
        )
    elif mode == 'fm':
        return pipeline.inference_fm(
            src_wav_path=content,
            timbre_ref_wav_path=ref_timbre,
            flow_matching_steps=steps
        )
    elif mode == 'tts':
        print(src_language)
        print(ref_language)
        return pipeline.inference_ar_and_fm(
            task="synthesis",
            src_wav_path=None,
            src_text=src_text,
            style_ref_wav_path=ref_style,
            timbre_ref_wav_path=ref_timbre,
            style_ref_wav_text=ref_text if ref_text != '' else None,
            src_text_language=src_language,
            style_ref_wav_text_language=ref_language,
            flow_matching_steps=steps
        )
    else:
        raise RuntimeError("Unrecognized inference mode '{}'".format(mode))

def infer():
    try:
        content_filename = content_path.get()
        reference_style_filename = reference_style_path.get()
        reference_timbre_filename = reference_timbre_path.get()
        output_dir = output_path.get()
        if not os.path.isdir(output_dir):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), output_dir)
        if not os.path.isfile(content_filename):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), content_filename)
        if not os.path.isfile(reference_style_filename):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), reference_style_filename)
        if not os.path.isfile(reference_timbre_filename):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), reference_timbre_filename)
        
        output_filename = get_unique_filename(os.path.join(output_dir, 'output'), 'wav')
        gen_audio = run_inference(
            inference_pipeline,
            mode = mode_var.get(),
            content = content_filename,
            ref_style = reference_style_filename,
            ref_timbre = reference_timbre_filename,
            src_text = source_text.get(),
            ref_text = reference_text.get(),
            src_language = source_language.get(),
            ref_language = reference_language.get(),
            steps = steps_value.get())
        vevosing_utils.save_audio(gen_audio, target_sample_rate=48000, output_path=output_filename)
        message = "Done. Output file: '{}'".format(output_filename)
        print(message)
        error_str.set(message)
    except Exception as e:
        error_str.set(str(e))
        traceback.print_exc()

def load_model():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Content Tokenizer
    local_dir = snapshot_download(
        repo_id="amphion/Vevo1.5",
        repo_type="model",
        cache_dir="./ckpts/Vevo1.5",
        allow_patterns=["tokenizer/prosody_fvq512_6.25hz/*"],
    )
    prosody_tokenizer_ckpt_path = os.path.join(
        local_dir, "tokenizer/prosody_fvq512_6.25hz"
    )

    # Content-Style Tokenizer
    local_dir = snapshot_download(
        repo_id="amphion/Vevo1.5",
        repo_type="model",
        cache_dir="./ckpts/Vevo1.5",
        allow_patterns=["tokenizer/contentstyle_fvq16384_12.5hz/*"],
    )
    content_style_tokenizer_ckpt_path = os.path.join( local_dir, "tokenizer/contentstyle_fvq16384_12.5hz")

    # Autoregressive Transformer
    ar_model_name = "ar_emilia101k_singnet7k"
    local_dir = snapshot_download(
        repo_id="amphion/Vevo1.5",
        repo_type="model",
        cache_dir="./ckpts/Vevo1.5",
        allow_patterns=[f"contentstyle_modeling/{ar_model_name}/*"],
    )
    ar_cfg_path = f"./config/{ar_model_name}.json"
    ar_ckpt_path = os.path.join(local_dir, "contentstyle_modeling/ar_emilia101k_singnet7k")

    # Flow Matching Transformer
    fm_model_name = "fm_emilia101k_singnet7k"
    local_dir = snapshot_download(
        repo_id="amphion/Vevo1.5",
        repo_type="model",
        cache_dir="./ckpts/Vevo1.5",
        allow_patterns=[f"acoustic_modeling/{fm_model_name}/*"],
    )
    fmt_cfg_path = f"./config/{fm_model_name}.json"
    fmt_ckpt_path = os.path.join(local_dir, "acoustic_modeling/fm_emilia101k_singnet7k")

    # Vocoder
    local_dir = snapshot_download(
        repo_id="amphion/Vevo1.5",
        repo_type="model",
        cache_dir="./ckpts/Vevo1.5",
        allow_patterns=["acoustic_modeling/Vocoder/*"],
    )
    vocoder_cfg_path = "./Amphion/models/svc/vevosing/config/vocoder.json"
    vocoder_ckpt_path = os.path.join(local_dir, "acoustic_modeling/Vocoder")

    # Inference
    pipeline = vevosing_utils.VevosingInferencePipeline(
        prosody_tokenizer_ckpt_path=prosody_tokenizer_ckpt_path,
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

def transcribe(filename):
    print('Transcribing...')
    import whisper
    whisper_model = whisper.load_model("large-v3-turbo", device="cuda", download_root="./ckpts/")
    result = whisper_model.transcribe(filename)
    #print(result)
    return result["text"], result["language"]

def browse_reference_style():
    try:
        filename = filedialog.askopenfilename(filetypes=(("Audio files","*.wav *.mp3"),("All files","*.*")))
        if os.path.exists(filename):
            reference_style_path.set(get_wav(filename, output_path.get()))
            if reference_transcribe_checked.get() == 1:
                text, language = transcribe(filename)
                reference_text.set(text)
                if language == 'en':
                    reference_language_combo.current(0)
                elif language == 'zh':
                    reference_language_combo.current(1)

    except Exception as e:
        messagebox.showerror('Error', 'Tried to generate .wav file in output directory, but failed: {}'.format(e))

def browse_reference_timbre():
    try:
        filename = filedialog.askopenfilename(filetypes=(("Audio files","*.wav *.mp3"),("All files","*.*")))
        if os.path.exists(filename):
            reference_timbre_path.set(get_wav(filename, output_path.get()))
    except Exception as e:
        messagebox.showerror('Error', 'Tried to generate .wav file in output directory, but failed: {}'.format(e))

def browse_content():
    try:
        filename = filedialog.askopenfilename(filetypes=(("Audio files","*.wav *.mp3"),("All files","*.*")))
        if os.path.exists(filename):
            content_path.set(get_wav(filename, output_path.get()))
            if source_transcribe_checked.get() == 1 and mode_var.get() != 'fm':
                text, language = transcribe(filename)
                source_text.set(text)
                if language == 'en':
                    source_language_combo.current(0)
                elif language == 'zh':
                    source_language_combo.current(1)

    except Exception as e:
        messagebox.showerror('Error', 'Tried to generate .wav file in output directory, but failed: {}'.format(e))

def browse_output():
    dirname = filedialog.askdirectory()
    if not os.path.isdir(dirname):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), dirname)
    output_path.set(dirname)

def set_mode():
    # Set certain controls to enabled or disabled depending on the inference mode
    mode = mode_var.get()
    if mode == 'style':
        reference_style_entry['state'] = tk.NORMAL
        reference_style_browse['state'] = tk.NORMAL
        reference_timbre_entry['state'] = tk.NORMAL
        reference_timbre_browse['state'] = tk.NORMAL
        content_entry['state'] = tk.NORMAL
        content_browse['state'] = tk.NORMAL
        source_text_entry['state'] = tk.NORMAL
        source_language_combo['state'] = tk.NORMAL
        reference_text_entry['state'] = tk.NORMAL
        reference_language_combo['state'] = tk.NORMAL
        source_transcribe_checkbutton['state'] = tk.NORMAL
        reference_transcribe_checkbutton['state'] = tk.NORMAL
    elif mode == 'fm':
        reference_style_entry['state'] = tk.DISABLED
        reference_style_browse['state'] = tk.DISABLED
        reference_timbre_entry['state'] = tk.NORMAL
        reference_timbre_browse['state'] = tk.NORMAL
        content_entry['state'] = tk.NORMAL
        content_browse['state'] = tk.NORMAL
        source_text_entry['state'] = tk.DISABLED
        source_language_combo['state'] = tk.DISABLED
        reference_text_entry['state'] = tk.DISABLED
        reference_language_combo['state'] = tk.DISABLED
        source_transcribe_checkbutton['state'] = tk.DISABLED
        reference_transcribe_checkbutton['state'] = tk.DISABLED
    elif mode == 'tts':
        reference_style_entry['state'] = tk.NORMAL
        reference_style_browse['state'] = tk.NORMAL
        reference_timbre_entry['state'] = tk.NORMAL
        reference_timbre_browse['state'] = tk.NORMAL
        content_entry['state'] = tk.DISABLED
        content_browse['state'] = tk.DISABLED
        source_text_entry['state'] = tk.NORMAL
        source_language_combo['state'] = tk.NORMAL
        reference_text_entry['state'] = tk.NORMAL
        reference_language_combo['state'] = tk.NORMAL
        source_transcribe_checkbutton['state'] = tk.DISABLED
        reference_transcribe_checkbutton['state'] = tk.NORMAL
        
    else:
        pass

if __name__ == '__main__':
    inference_pipeline = load_model()
    root = tk.Tk()
    root.title('Vevo GUI')
    root.geometry("1200x300")

    tk.Grid.columnconfigure(root, 1, weight=1) # Weight this column to have it stretch with the window

    # Voice mode options
    reference_style_label = tk.Label(root, text='Reference style:')
    reference_style_path = tk.StringVar()
    reference_style_path.set('./samples/ian-mckellan.wav')
    reference_style_entry = tk.Entry(root, textvariable=reference_style_path)
    reference_style_browse = tk.Button(root, text='Browse', command=browse_reference_style)
    reference_timbre_label = tk.Label(root, text='Reference timbre:')
    reference_timbre_path = tk.StringVar()
    reference_timbre_path.set('./samples/ian-mckellan.wav')
    reference_timbre_entry = tk.Entry(root, textvariable=reference_timbre_path)
    reference_timbre_browse = tk.Button(root, text='Browse', command=browse_reference_timbre)
    content_label = tk.Label(root, text='Source audio/melody:')
    content_path = tk.StringVar()
    content_path.set('./samples/barry-white.wav')
    content_entry = tk.Entry(root, textvariable=content_path)
    content_browse = tk.Button(root, text='Browse', command=browse_content)

    # TTS mode options
    source_text = tk.StringVar()
    source_text_label = tk.Label(root, text='Source text:')
    source_text_entry = tk.Entry(root, textvariable=source_text)
    source_text.set("I just want some. Someone to talk to. I want you just the way you are.")
    source_transcribe_checked = tk.IntVar()
    source_transcribe_checkbutton = tk.Checkbutton(root, text='Auto Transcribe',variable=source_transcribe_checked, onvalue=1, offvalue=0)
    #source_transcribe_checkbutton.select()
    reference_text = tk.StringVar()
    reference_text_label = tk.Label(root, text='Reference style text:')
    reference_text_entry = tk.Entry(root, textvariable=reference_text)
    reference_text.set("It is an ancient mariner, and he stoppeth one of three. By thy long grey beard and thy glittering eye, now wherefore stoppest me. The bridegroom's doors are opened wide and I am next of kin.")
    reference_transcribe_checked = tk.IntVar()
    reference_transcribe_checkbutton = tk.Checkbutton(root, text='Auto Transcribe',variable=reference_transcribe_checked, onvalue=1, offvalue=0)
    #reference_transcribe_checkbutton.select()
    language_options = ('en', 'zh')
    source_language = tk.StringVar()
    source_language_combo = ttk.Combobox(root, textvariable=source_language, width=8)
    source_language_combo['values'] = language_options
    source_language_combo.current(0)
    reference_language = tk.StringVar()
    reference_language_combo = ttk.Combobox(root, textvariable=reference_language, width=8)
    reference_language_combo['values'] = language_options
    reference_language_combo.current(0)

    # Output directory
    outdir = './output'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    output_label = tk.Label(root, text='Output Directory:')
    output_path = tk.StringVar()
    output_path.set(outdir)
    output_entry = tk.Entry(root, textvariable=output_path)
    output_browse = tk.Button(root, text='Browse', command=browse_output)

    # Number of inference steps
    steps_value = tk.IntVar()
    steps_value.set(32)
    steps_label = tk.Label(root, text='Flow Matching Steps:')
    steps_scale = tk.Scale(root, variable= steps_value, from_=1, to=64, orient=tk.HORIZONTAL)
    
    # Radio buttons for inference mode
    mode_label = tk.Label(root, text='Inference mode:')
    mode_frame = tk.Frame(root)
    
    mode_button_dict = {
        'Style (ar and fm recognition-synthesis)'  : 'style',
        'Timbre (fm)': 'fm',
        'TTS (ar and fm synthesis)'   : 'tts'
    }
    mode_var = tk.StringVar()
    col = 0
    for (text, value) in mode_button_dict.items():
        rb = tk.Radiobutton(mode_frame, text = text, variable = mode_var, command=set_mode, value = value)
        rb.grid(row=0,column=col,sticky=tk.EW)
        mode_frame.columnconfigure(col,weight=1)
        col += 1
    mode_var.set('fm')

    infer_button = tk.Button(root, text='Run Inference', command=infer)

    error_str = tk.StringVar()
    error_label = tk.Label(root, textvariable=error_str)

    set_mode()
    
    reference_style_label.grid(row=0,column=0)
    reference_style_entry.grid(row=0,column=1,sticky=tk.EW)
    reference_style_browse.grid(row=0,column=2)
    reference_transcribe_checkbutton.grid(row=0,column=3)
    reference_text_label.grid(row=1,column=0)
    reference_text_entry.grid(row=1,column=1, columnspan=2, sticky=tk.EW)
    reference_language_combo.grid(row=1,column=3)
    
    reference_timbre_label.grid(row=2,column=0)
    reference_timbre_entry.grid(row=2,column=1,sticky=tk.EW)
    reference_timbre_browse.grid(row=2,column=2)
    
    content_label.grid(row=3,column=0)
    content_entry.grid(row=3,column=1,sticky=tk.EW)
    content_browse.grid(row=3,column=2)
    source_transcribe_checkbutton.grid(row=3, column=3)
    
    source_text_label.grid(row=4,column=0)
    source_text_entry.grid(row=4,column=1, columnspan=2, sticky=tk.EW)
    source_language_combo.grid(row=4,column=3)

    output_label.grid(row=5,column=0)
    output_entry.grid(row=5,column=1,sticky=tk.EW)
    output_browse.grid(row=5,column=2)
    
    steps_label.grid(row=6,column=0,sticky=tk.NSEW)
    steps_scale.grid(row=6,column=1,sticky=tk.EW)
    
    mode_label.grid(row=7,column=0,sticky=tk.NSEW)
    mode_frame.grid(row=7,column=1,sticky=tk.EW)
    
    infer_button.grid(row=8,column=1)
    error_label.grid(row=9,column=1)
    root.mainloop()

