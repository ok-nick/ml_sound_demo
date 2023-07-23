# `ml_sound_demo`
Machine learning demonstration using Whisper for audio transcription and Wav2Vec2 for audio gender classification.

## Project Structure
```
ml_sound_demo/
├── flake.lock                      # nix flake lock file for deterministic builds
├── flake.nix                       # dev environment flake for the nix package manager
├── LICENSE.md
├── ml_sound_demo
│   ├── gender.py                   # dataset gender classification interface
│   ├── __init__.py                 # export public interface
│   └── transcribe.py               # dataset transcription interface
├── output
│   ├── gender_classification_model # trained gender classification model
│   ├── transcription_100.csv       # first 100 transcriptions in csv
│   └── transcription_100.txt       # first 100 transcriptions in txt
├── pyproject.toml                  # project manifest
├── README.md                       # you know...
├── tests
│   ├── gender_1000.py              # gender of 1000 audio clips in common voice 13.0
│   ├── __init__.py
│   └── transcribe_100.py           # transcription of 100 audio clips in common voice 13.0
└── training
    └── gender.py                   # gender classification model training
```

## Outstanding Questions
* How are projects similar to this usually structured? File structure? API structure?
* Which libraries are *actually* needed/recommended?
* How can I make the code better/more optimal? Which functions to use? Which properties to tweak?
    * Conventions?
