# Vevo GUI
Simple GUI for Amphion Vevo: https://github.com/open-mmlab/Amphion

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
