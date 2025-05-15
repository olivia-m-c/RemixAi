# Project Title

RemixAI: Automatic Instrument Separation and MIDI/XML Transcription

## Table of Contents
- [Introduction](#introduction)
- [How to Run](#how-to-run)
- [Requirements](#requirements)
- [Libraries](#libraries)
- [Project Details](#project-details)
- [Results](#results)
- [Author](#author)


## Introduction

This Project is the first draft of an app which final aim is to transform a song into different genres. These would include changing rhythmic patterns, instruments, even keys. This draft converts an initial .mp3 file into different .mid and .xml files and separates the instruments.

## How to Run

1. Install dependencies. Also, recommended: python 3.8
2. Run the script

### Requirements

- Python 3.8
- demucs
- pretty_midi
- librosa
- basic_pitch
- music21
- numpy
- scipy
- torch
- torchaudio
- matplotlib
- midi2audio




#### Libraries

The libraries used were:
—	**Librosa**(to get audio files, extract tempo information)
—	**Pretty midi** (to work with the .mid files: creating a new midi file, reading them, writing them, accessing instrument information)
—	**Basic pitch**: to convert the initial .mp3 file into a .mid file. Basic pitch is a that uses machine learning and AI to convert audio into MIDI. It's specifically designed for polyphonic pitch detection (meaning it can detect multiple notes being played simultaneously in an audio file). It is a deep learning model (specifically a neural network) that was trained to recognize musical notes in audio. 
—	**Demucs**: deep learning model used to separate instruments in different wav files. The model is called: ‘htdemucs_6s’. It takes the full audio file mix as input, and then using the neural network it identifies and isolate different instruments, outputting separate audio files (wav) for each instrument. 
—	**Music21**: toolkit to convert midi files to xml files
—	**Numpy**: used mainly for:
    o	Handling audio data arrays (since audio is essentially numerical data)
    o	Calculating audio features like RMS (Root Mean Square)
    o	Working with tempo calculations
    o	Converting between different data formats
—	**Scipy** is used for signal processing, in particular, to create and apply bandpass filters to audio signals. Necessary to isolate and properly process the different drum sounds by frequency.

##### Project Details

In this repository, I added an initial mp3 demo I created based on the chord progression of For No One (The Beatles) with: guitar, keys, drums, and bass. The name of this file in the script is: trial_ensemble.mp3.

Currently, this is how the script works: 

Version 1: Draft.  In this one I worked on the proper transcription of a song. The input is a .mp3 or wav, and the outputs are individual .mid files and  .xml files of each instrument. The script takes the song (which, since this was a first draft, is automatically inserted in the script) in this case is: trial_ensemble.mp3., identifies the instruments and separates them into wavs, and then converts them into .mid and .xml files. These are the steps you can follow in the main: 

1.	**Set up**: get the tempo of the song in its wav/mp3 format, name the file properly.

2.	**Instrument separation**: Separate the initial wav into its different instruments (creates a file in the directory: separated_wavs). Basic pitch only creates midis with a piano sound, so by separating the wavs it was easier to properly identify the instrument and tell the program so pretty midi could create a midi with the actual instrument after basic pitch had created the midi. 

3.	**MIDI Conversion**: Converting the separate wavs into .midi. Basic_pitch initially converts the audio to MIDI notes (it doesn't assign instruments). The code then determines the instrument based on the filename, and pretty_midi sets the correct instrument program. The drums are converted with a different script developed (drumtest_1_1.py) as it was a bit of a more complex process. It detects the different drum hits (kick,snare, hihats, toms, cymbal..) and converts those hits into a MIDI file. It also provides tools to visualize and analyse the audio and it is reflected in the terminal so the developer can be oriented (uses scipy and matplotlib)

4.	**MusicXML Conversion**: Finally,  the script converts each MIDI file to MusicXML. Also, it creates a combined_midi file with all the midis together. 

5.	**Transcription Assessment**: The transcription is properly done: mp3/wavs are properly converted into .mid files and .xml (even though they 'won't be necessary for this project). The next step, where the project is currently at, is changing the instruments used. This is why in the script it asks which instrument from the original mix you’d like to change and into what instrument. 

Note: The next step that is being developed is **Changing instruments**. It has been commented out as it is not finished, so the program runs smoothly. The functions choose_instrument and changing_piano_to_guitar (unfinished!) are being written for this task (in the main, lines 626, 627)

##### Results

The files created are: 

**Separated wavs of each instrument**
**conversion: midi and xml files of each instrument**
**combined midi file of all the instruments**
**"MODIFIED MIDI!** Does not work yet

The .mid files created are acceptable. It gets the tempo and rhythm accurately enough. It should be fine-tuned, as it doesn’t sound as the original yet. And it does not sound human yet. But it is good enough to work with it and use it for the final purpose of this project. Even though the .xml, when reproduced by the machine, keeps the tempo and rhythm, it is not acceptable. The rhythm and notes are not written in the way a human musician would read it easily. If it was to be used, it should be cleaned up and improved for practical use. It was left like this since we probably don’t need it to change instruments and genres, it made more sense that it is modified through the .mid files. The next developement step is to enable changing instruments and rhythms, since the transcription assessment has been successful.
The folder with the name "modified midi" is the next step, where the user of the script can choose (using the terminal) the original instrument they'd like to change and to which instrument they'd like to change it to.

##Author 
- GitHub: [@olivia-m-c](https://github.com/olivia-m-c)
