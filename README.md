# Vevo GUI
Simple GUI for Amphion Vevo: https://github.com/open-mmlab/Amphion

Updated to support [Vevo 1.5](https://huggingface.co/amphion/Vevo1.5) a.k.a. [vevosing](https://github.com/open-mmlab/Amphion/blob/main/models/svc/vevosing/README.md)

## Installation
- Clone this repository and `cd` to the top level.
- Create and activate python virtual environment:
  - `python -m venv venv`
  - `source venv/bin/activate` or `venv\Scripts\activate.bat`
- Install pytorch using the recommended command from the website:
  - https://pytorch.org/
- Install the dependencies:
  - `pip install -r requirements.txt`
- Run the app (current working directory needs to be the top level of the repo):
  - `python app.py`
- This was tested on Linux+AMD (ROCM 6.2.4) and Python 3.10. It should work for Nvidia and Python 3.13.

### Pinokio Install
In Pinokio, select "Download from URL" and use this:\
https://github.com/chameleon-ai/vevo-pinokio

# Vevo 1
Run `vevo_gui.py` or `app.py -v 1`

![screen2](https://github.com/user-attachments/assets/c5a3ee3d-dc79-4bbd-bf1f-cfc72ec21fba)

## Inference Modes
### Voice
A combo of vevostyle and vevovoice. Capable of imitating a voice and style (accent, emotion) independently.\
The only difference is that vevostyle uses the same audo for source and timbre, and vevovoice uses the same audio for style and timbre.
- https://github.com/open-mmlab/Amphion/blob/main/models/vc/vevo/infer_vevostyle.py
- https://github.com/open-mmlab/Amphion/blob/main/models/vc/vevo/infer_vevovoice.py
### Timbre
A direct port of vivotimbre. A more direct voice conversion, also capable of processing longer audio clips than vevostyle/vevovoice.
- https://github.com/open-mmlab/Amphion/blob/main/models/vc/vevo/infer_vevotimbre.py
### TTS
A direct port of vevotts. Doesn't work very well.
- https://github.com/open-mmlab/Amphion/blob/main/models/vc/vevo/infer_vevotts.py

# Vevo 1.5 (vevosing)
Run `vevosing_gui.py` or `app.py -v 1.5`

![Image](https://github.com/user-attachments/assets/d69bfc2b-f427-402e-8ee1-1297f2ab115b)

## Auto Transcription
Vevo 1.5 requires accurate transcripts of the reference audio when using some modes. When the `Auto Transcribe` checkbox is checked, the audio will automatically be transcribed on selection using [openai-whisper](https://github.com/openai/whisper), specifically the [large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) model.

It also attempts to automatically detect and select the audio language. As far as I'm aware, only English and Chinese are supported.

**Enabling transcription will bump up the VRAM usage. This is a completely opt-in feature.**\
If the model is too much for you, edit the `transcribe` function in vevosing_gui.py:
````
def transcribe(filename):
    print('Transcribing...')
    import whisper
    whisper_model = whisper.load_model("large-v3-turbo", device="cuda", download_root="./ckpts/")
````
and try "medium" or "tiny".

## Length Limit
Vevo 1.5 is designed for short clips (about 15 seconds or less). The GUI won't stop you from using longer clips, but be warned that it may not work.

If you see this warning, select 'Yes' to truncate the clip at 15 seconds. The truncated file will be put in the output directory and used as input. Select 'No' to keep the full length audio.

![Image](https://github.com/user-attachments/assets/8b3b0be1-f530-4684-862e-463ae6b46ba4)

## Inference Modes
### Style
A combo of `vevosing_editing`, `vevosing_singing_style_conversion` and `vevosing_melody_control`.\
The only differences between these are how the inputs are used.
- `vevosing_editing` uses the same input for reference style, reference timbre, and source audio. The reference style text should be an accurate transcript of the reference style audio. It will output whatever the source text says.
- `vevosing_singing_style_conversion` uses the same input for reference timbre and source audio. The source text and reference style text should be accurate transcripts of their respective audio inputs. It basically only converts the style / accent of the reference style.
- `vevosing_melody_control` uses the same input for reference timbre and reference style.  The reference style text should be an accurate transcript of the reference style audio. The source audio can be any melody, such as a piano. It will output whatever the source text says.
- Really, it's all just a call to the same `inference_pipeline.inference_ar_and_fm` function, so just experiment with different combinations of inputs.
- https://github.com/open-mmlab/Amphion/blob/main/models/svc/vevosing/infer_vevosing_ar.py
### Timbre
A direct port of `vevosing_fm`. Doesn't use a transcript. Works on clips about 15 seconds long.
- https://github.com/open-mmlab/Amphion/blob/main/models/svc/vevosing/infer_vevosing_fm.py
### TTS
A direct port of `vevsing_tts`. This mode basically does arbitrary text-to-speech based on the voice cloned from input audio.
- https://github.com/open-mmlab/Amphion/blob/main/models/svc/vevosing/infer_vevosing_ar.py
