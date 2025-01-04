# Vevo GUI
Simple GUI for Amphion Vevo: https://github.com/open-mmlab/Amphion

![screen2](https://github.com/user-attachments/assets/c5a3ee3d-dc79-4bbd-bf1f-cfc72ec21fba)

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
- This is confirmed working for Linux+AMD (ROCM 6.2). If you run into issues, please try to get the command-line example from the Amphion repo working first. This GUI is a simple wrapper for the Amphion tools and there is no platform specific code in the GUI.
  - https://github.com/open-mmlab/Amphion/blob/main/models/vc/vevo/README.md

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
