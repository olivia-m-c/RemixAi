import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model
import os
import librosa
import pretty_midi
import numpy as np
from basic_pitch.inference import predict
import music21 #import converter, stream
import drumtest_1_1
#from drumstest_1_1 import wav_to_drum_midi
from music21 import *
import glob
from scipy.io import wavfile
import traceback

def separate_instruments(file_path):
    
    # Print current working directory and check if file exists
    print(f"Current working directory: {os.getcwd()}")

    print(f"File exists: {os.path.exists(file_path)}")
    song_name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"Song name without extension: {song_name}")

    # Load model
    model = get_model('htdemucs_6s')

    # Check if MPS is available (for M1/M2 Macs)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        model.to(device)
        print("Using MPS backend")
    else:
        # Fallback to CPU
        device = torch.device("cpu")
        model.cpu()
        print("Using CPU backend")

    try:
        # Load audio using librosa
        print("Loading audio file...")
        audio_numpy, sr = librosa.load(file_path, sr=44100, mono=False)
        
        # Convert to torch tensor and reshape properly
        input_audio = torch.tensor(audio_numpy)
        
        # Ensure correct shape (batch, channels, length)
        if input_audio.dim() == 1:
            # If mono, add channel and batch dimensions
            input_audio = input_audio.unsqueeze(0).unsqueeze(0)
        elif input_audio.dim() == 2:
            # If stereo, just add batch dimension
            input_audio = input_audio.unsqueeze(0)
        
        print(f"Input audio tensor shape: {input_audio.shape}")
        print(f"Successfully loaded audio file with sample rate: {sr}")
        
        if torch.backends.mps.is_available():
            input_audio = input_audio.to(device)

        # Separate
        sources = apply_model(model, input_audio, progress=True)
        
        # Convert sources to proper format
        sources = sources.cpu()
        
        # Print shape of sources
        print(f"Sources shape: {sources.shape}")
        print(f"Sources: {sources[0][1]}")
        # Sources shape is [1, 4, 2, samples]
        # Remove batch dimension and get each stem
        sources = sources.squeeze(0)  # Remove batch dimension, now [4, 2, samples]
        
        # if the folder does not exist, create it
        output_dir = "separated_wavs"
        os.makedirs(output_dir,exist_ok=True)
        
        # Save separated tracks
        for idx, source_name in enumerate(model.sources):
            print(f"Processing {source_name}")
            source_audio = sources[idx]
            print(f"Shape for {source_name}: {source_audio.shape}")
            # Path to save the separated wav files
            output_path = os.path.join(output_dir, f'{song_name}_{source_name}.wav')
            torchaudio.save(output_path, source_audio, sr)
            print(f"Saved {output_path}")
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()



def mp3_to_midi(mp3_path, midi_output_path):
    

    """Convert MP3 / WAV to MIDI using basic_pitch with multi-instrument support"""
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict_and_save
    print("debug mp3_path name: ", mp3_path) #flag

    try:
         # Load WAV file, verify it has significant content
        y, sr = librosa.load(mp3_path, sr=44100)
        # Calculation of RMS
        rms = librosa.feature.rms(y=y)[0]

        average_rms = np.mean(rms)
        print(f"Average RMS for {mp3_path}: {average_rms}")
        # If RMS too low, file considered empty
        if average_rms < 0.01:
            print(f"File {mp3_path} is empty or has no significant content. Skipping...")
            return None #no returns any file, not even an empty one

        print(f"Processing instrument: {os.path.basename(mp3_path)}")    #debug 

        instrument_name = os.path.basename(mp3_path).split('_')[2].split('.')[0] 
        print(f"Processing instrument: {instrument_name}")  # Debug
        
        midi_output_path_drums ="midi_files/trial_ensemble_drums.mid"
        # MIDI program according to instrument
        if 'guitar' in instrument_name:
            program = 25  # Electric Guitar
        elif 'bass' in instrument_name:
            program = 33  # Electric Bass
        elif 'piano' in instrument_name:
            program = 0   # Acoustic Piano
        elif 'drums' in instrument_name:
            program = 10  # Drums. We're never going to use this, its just so it does not crash due to no value in program var
            
        elif 'other' in instrument_name:
            program = 0   # Default to piano for 'other'
        else:
            program = 0   # Default to piano
        
        print(f"Selected program: {program}")  # Debug
        if (program == 10):
            print("Drums detected, converting wav into MIDI...")
            drum_counts = drumtest_1_1.wav_to_drum_midi(mp3_path, midi_output_path)
            print("Drums successfully converted into MIDI...")
        else: #the machine "listens" to the audio and "understands" it so it and detects instruments, chords, notes,rhythms..etc
            predict_and_save(
                model_or_model_path=ICASSP_2022_MODEL_PATH,
                audio_path_list=[mp3_path],
                output_directory=os.path.dirname(midi_output_path),
                save_midi=True,
                sonify_midi=False,
                save_model_outputs=False,
                save_notes=False
            )
            basic_pitch_midi = midi_output_path.replace('.mid', '_basic_pitch.mid')
            if os.path.exists(basic_pitch_midi):
                pm = pretty_midi.PrettyMIDI(basic_pitch_midi)
                print(f"Before change - Program: {pm.instruments[0].program}")  # Debug
                
                for instrument in pm.instruments:
                    for note in instrument.notes:
                        duration = note.end - note.start
                        note.end = note.start + (duration / 2)  # Make each note half as long (for better accuracy)
                
                # Change instrument program
                for instrument in pm.instruments:
                    instrument.program = program
                    if 'drums' in instrument_name:
                        instrument.is_drum = True
                        print("Setting as drums")  # Debug
                
                print(f"After change - Program: {pm.instruments[0].program}")  # Debug
                
                # debug. calculating the tempo
                estimated_tempo = pm.estimate_tempo()
                print(f"Estimated tempo: {estimated_tempo} BPM of instrument {instrument_name}")
                # save modified midi
                pm.write(midi_output_path)
                # remove temporary midi
                os.remove(basic_pitch_midi)
                return midi_output_path
            
    except Exception as e:
        print(f"Error processing {mp3_path}: {str(e)}")
        return None

def changing_piano_to_guitar(midi_path, midi_output_path):
    #unfinished
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)

        for i,instrument in enumerate(pm.instruments): #debug, checking instruments
            print(f"instrument {i+1}: {instrument.name} and program: {instrument.program}")
        for instruments in pm.instruments:
            #if 'piano' in instruments.name: #another way of doing it 
            if instrument.program == 0:
                instrument.program = 25 #guitar . bass is 33
                print(f"changing piano to guitar: {instrument.program}")
        
        # checking if the directory exists
        os.makedirs(os.path.dirname(midi_output_path), exist_ok=True)
        
        # save file + return
        pm.write(midi_output_path)
        return midi_output_path
    
    except Exception as e:
        print(f"Error changing piano to guitar: {str(e)}")
        traceback.print_exc()
        return None

def midi_to_musicxml(midi_path, xml_output_path, tempo):
    """Convert MIDI to MusicXML with better instrument and voice separation"""
    from music21 import meter, key, instrument, converter, stream, tempo as m21_tempo, metadata, expressions, clef
    
    # Load MIDI file
    pm = pretty_midi.PrettyMIDI(midi_path)
    
    # Load MIDI file with music21 (for the notes/structure)
    midi_stream = converter.parse(midi_path)
    
    # Create main score
    score = stream.Score()
    
    # Add metadata
    score.insert(0, metadata.Metadata())
    
    # Add time signature
    time_signatures = pm.time_signature_changes
    if time_signatures:
        time_sig = meter.TimeSignature(f'{time_signatures[0].numerator}/{time_signatures[0].denominator}')
    else:
        time_sig = meter.TimeSignature('4/4')
    
    # Try to get key signature (might fail for drum tracks)
    try:
        key_analysis = midi_stream.analyze('key')
    except:
        # Default to C major if key analysis fails, for drums
        key_analysis = key.Key('C')
    
    # Create tempo mark
    mm = m21_tempo.MetronomeMark(number=tempo)
    first_part_created = False
    
    # For piano, create two parts: right hand (treble) and left hand (bass)
    for midi_instrument in pm.instruments:
        if len(midi_instrument.notes) == 0:
            continue
            
        # Create a new part
        part = stream.Part()
        
        # Detect instrument type
        if midi_instrument.program in range(0, 8):  # Piano sounds
            # Create right hand part (treble)
            treble_part = stream.Part()
            treble_part.append(clef.TrebleClef())
            if not first_part_created:
                treble_part.insert(0, mm)  # Add tempo to first part
                first_part_created = True
            treble_part.append(time_sig)
            treble_part.append(key_analysis)
            treble_part.append(instrument.Piano())
            
            # Create left hand part (bass)
            bass_part = stream.Part()
            bass_part.append(clef.BassClef())
            bass_part.append(time_sig)
            bass_part.append(key_analysis)
            bass_part.append(instrument.Piano())
            
            # Split notes between hands
            MIDDLE_C = 60
            for midi_note in midi_instrument.notes:
                n = music21.note.Note(
                    pitch=midi_note.pitch,
                    quarterLength=round((midi_note.end - midi_note.start) * 4) / 4
                )
                if midi_note.pitch >= MIDDLE_C:
                    treble_part.insert(round(midi_note.start * 4) / 4, n)
                else:
                    bass_part.insert(round(midi_note.start * 4) / 4, n)
            
            treble_cleaned = treble_part.makeNotation()
            bass_cleaned = bass_part.makeNotation()
            score.append(treble_cleaned)
            score.append(bass_cleaned)
            
        elif midi_instrument.program in range(32, 40):  # Bass sounds
            part.append(clef.BassClef())
            if not first_part_created:
                part.insert(0, mm)  # Add tempo to first part
                first_part_created = True
            part.append(time_sig)
            part.append(key_analysis)
            part.append(instrument.ElectricBass())
            
            for midi_note in midi_instrument.notes:
                n = music21.note.Note(
                    pitch=midi_note.pitch,
                    quarterLength=round((midi_note.end - midi_note.start) * 4) / 4
                )
                part.insert(round(midi_note.start * 4) / 4, n)
                
        elif midi_instrument.program in range(24, 32):  # Guitar sounds
            part.append(clef.TrebleClef())
            if not first_part_created:
                part.insert(0, mm)  # Add tempo to first part
                first_part_created = True
            part.append(time_sig)
            part.append(key_analysis)
            part.append(instrument.ElectricGuitar())
            
            for midi_note in midi_instrument.notes:
                n = music21.note.Note(
                    pitch=midi_note.pitch,
                    quarterLength=round((midi_note.end - midi_note.start) * 4) / 4
                )
                part.insert(round(midi_note.start * 4) / 4, n)
                
        elif midi_instrument.is_drum:  # Drums
            part.append(clef.PercussionClef())
            if not first_part_created:
                part.insert(0, mm)  # Add tempo to first part
                first_part_created = True
            part.append(time_sig)
            part.append(key_analysis)
            part.append(instrument.Percussion())
            
            for midi_note in midi_instrument.notes:
                n = music21.note.Note(
                    pitch=midi_note.pitch,
                    quarterLength=round((midi_note.end - midi_note.start) * 4) / 4
                )
                part.insert(round(midi_note.start * 4) / 4, n)
                
        else:  # Other instruments
            part.append(clef.TrebleClef())
            if not first_part_created:
                part.insert(0, mm)  # Add tempo to first part
                first_part_created = True
            part.append(time_sig)
            part.append(key_analysis)
            inst = instrument.instrumentFromMidiProgram(midi_instrument.program)
            part.append(inst)
            
            for midi_note in midi_instrument.notes:
                n = music21.note.Note(
                    pitch=midi_note.pitch,
                    quarterLength=round((midi_note.end - midi_note.start) * 4) / 4
                )
                part.insert(round(midi_note.start * 4) / 4, n)
        
        # Clean up and add part (except for piano which is already added)
        if midi_instrument.program not in range(0, 8):
            part_cleaned = part.makeNotation()
            score.append(part_cleaned)
    
    # Write the score
    score.write('musicxml', fp=xml_output_path)

'''def midi_to_drum_score(midi_file):

    from music21 import meter, key, instrument, converter, stream, tempo as m21_tempo, metadata, expressions, clef
   

    # Load MIDI file with music21 (for the notes/structure)
    midi_stream = converter.parse(midi_file) #función de music21 que lee el archivo MIDI y lo convierte en un objeto STREAM de music21
    score = stream.Score()
    score.insert(0, metadata.Metadata())
    
    # Crear Part para drums
    part_drums = stream.Part()
    
    # Crear Measure
    measure = stream.Measure()
    
    # Configurar el measure
    time_signatures = midi_stream.getTimeSignatures()
    if time_signatures:
        time_sig = time_signatures[0]
    else:
        time_sig = meter.TimeSignature('4/4')
    measure.timeSignature = time_sig
    
    # Añadir elementos al measure
    measure.append(clef.PercussionClef())
    measure.append(key_analysis)
    measure.append(instrument.Percussion())
    
    # Añadir measure al part y part al score
    part_drums.append(measure)
    score.append(part_drums)
'''
def putting_midis_together(midi_paths, output_midis_path): #will have to prepare the midi_paths list beforehand
    #the object we're about to create only exists in the memory, not on the disk.
    combined_midi = pretty_midi.PrettyMIDI() #create a new pretty midi object empty.
    for midi_path in midi_paths: #example of midi_paths: "C:/Musica/proyecto/piano.mid"
        
        pm = pretty_midi.PrettyMIDI(midi_path)
        for instrument in pm.instruments:
            print(f"instrument in {midi_path}: {instrument.name}")
            combined_midi.instruments.append(instrument)
    #now we save the created object in the disk, as a file
    os.makedirs(os.path.dirname(output_midis_path), exist_ok=True)
    combined_midi.write(output_midis_path)

    return output_midis_path 

def midi_to_wav(midi_path, wav_output_path):
    #using sf: its a library of sounds(includes real audios of different instruments)(its like a sound bank that the synthesizer uses to convert MIDI notes into real audio)
    """Convert MIDI to WAV using FluidSynth with proper instrument sounds"""
    try:
        from midi2audio import FluidSynth

        # checking the content of the MIDI
        pm = pretty_midi.PrettyMIDI(midi_path)
        print(f"Number of instruments in the MIDI: {len(pm.instruments)}") #debug loop
        for i, instrument in enumerate(pm.instruments):
            print(f"Instrumento {i+1}: {len(instrument.notes)} notes")
        
        # checking the directory exists
        os.makedirs(os.path.dirname(wav_output_path), exist_ok=True)
        # update this path to the place where you saved the .sf2 file
        soundfont_path = "./GeneralUser-GS.sf2"
        
        # instance of FluidSynth with specific configuration
        fs = FluidSynth(sound_font=soundfont_path, sample_rate=44100)
        
        # Convertir con parámetros específicos
        fs.midi_to_audio(midi_path, wav_output_path)
        
        # check if the wav file was created
        if os.path.exists(wav_output_path):
            print(f"WAV file created successfully: {wav_output_path}")
        else:
            print("WAV file was not created!")
            
        return wav_output_path
        
    except Exception as e:
        print(f"Error in MIDI to WAV conversion: {str(e)}")
        import traceback 
        traceback.print_exc() #shows something similar to what would appear in the terminal
        return None

def convert_mp3_to_musicxml(mp3_path, tempo, output_dir=None):
    """Convert MP3 to MusicXML through MIDI intermediate"""
    output_dir = "midi_files" # RELATIVE PATH (doesnt start with /)
    print(f"the output directory is {output_dir}")
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output file paths
    song_name = os.path.splitext(os.path.basename(mp3_path))[0] 
    midi_path = os.path.join(output_dir, f"{song_name}.mid")
    print(f"the midi path is {midi_path}") #flag 
    xml_path = os.path.join(output_dir, f"{song_name}.xml")
    
    # Perform conversion mp3 to midi of each separated wav file
    wav_folder = "separated_wavs"
    for wav_file in os.listdir(wav_folder):
        if wav_file.startswith(f"{song_name}_") and wav_file.endswith(".wav"):
            wav_path = os.path.join(wav_folder, wav_file)
            print(f"wav_folder: {wav_folder}") #flag, erase
            print(f"wav_file: {wav_file}") #flag, erase
            print(f"wav_path: {wav_path}") 
            midi_name = wav_file.replace(".wav", ".mid")
            midi_path = os.path.join("midi_files", midi_name)
            print(f"midi_path: {midi_path}") 
            
            print(f"Converting {wav_file} to MIDI...")
            mp3_to_midi(wav_path, midi_path)
            print(f"Successfully converted {wav_file} to {midi_name}")
    #mp3_to_midi(mp3_path, midi_path)
    #midi_to_musicxml(midi_path, xml_path, tempo)

    # Convert each separated MIDI to a SINGLE MusicXML
    output_dir = "midi_files" 
    for midi_file in os.listdir(output_dir):
        if midi_file.startswith(f"{song_name}_") and midi_file.endswith(".mid"):
            midi_path = os.path.join(output_dir, midi_file)
            xml_name = midi_file.replace(".mid", ".xml")
            xml_path = os.path.join(output_dir, xml_name)

            pm = pretty_midi.PrettyMIDI(midi_path) #need this for the drums
            print("Converting MIDI to XML...")
            if any (instrument.is_drum for instrument in pm.instruments):
                print("Drums detected, will work on it on a separate function..")
                #midi_to_musicxml(midi_path, xml_path, tempo)
            else:
                midi_to_musicxml(midi_path, xml_path, tempo)
                
            print(f"Successfully converted {midi_file} to {xml_path}") #check what appears here
            
            print(f"\nSeeing what instruments the song has: {midi_file}:")
            for instrument in pm.instruments:
                # Determinate the instrument name based on the MIDI program
                if instrument.program in range(0, 8):
                    inst_name = "Piano"
                elif instrument.program in range(32, 40):
                    inst_name = "Bass"
                elif instrument.program in range(24, 32):
                    inst_name = "Guitar"
                else:
                    inst_name = f"Instrumento programa {instrument.program}"
                    
                print(f"- Instrumento: {inst_name}")
                print(f"- Es batería: {instrument.is_drum}")
                print(f"- Programa MIDI: {instrument.program}")
  
    
    #print(f"Successfully converted {mp3_path} to {output_path}")
    
    
    return xml_path # will return sth like "midi_files/The_Beatles_-_Help.xml" not a file! just the text that is the path

def get_tempo(audio_path):
    y, sr = librosa.load(audio_path) #y = actual audio data (numpy array, amplitude values of the signal)
    # sr: sampling rate (samples per second)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, start_bpm=120, units='time')
    #_ is to ignore the return value of the tuple from beat_track. it would be beat_frames: exact location where bets occur. we dont need it
    return tempo

def choose_instrument():
    print("\nChoose the instrument you want to change: ")
    print("1. Piano") #program number for piano is 0
    print("2. Guitar") #program number for guitar is 25
    print("3. Bass")
    print("4. Drums")
    print("5. Other")
    
    instrument_to_be_changed = 0
    instrument_final = 0
    
    while True:
        try:
            choice = int(input("Enter your choice (1-5): "))
            if choice == 1:
                instrument_to_be_changed = 0  # program number for piano
                break
            elif choice == 2:
                instrument_to_be_changed = 25  # Guitar
                break
            elif choice == 3:
                instrument_to_be_changed = 33  # Bass
                break
            elif choice == 4:
                instrument_to_be_changed = 10  # Drums
                break
            elif choice == 5:
                instrument_to_be_changed = 0  # Other
                break
            else:
                print("Invalid choice. Please enter a number between 1 and 5.")
        except ValueError:
            print("Please enter a valid number.")

    print("What instrument would you like to have in its place?")
    print("1. Piano")
    print("2. Guitar")
    print("3. Bass")
    print("4. Drums")
    print("5. Other")

    while True:
        try:
            choice = int(input("Enter your choice (1-5): "))
            if choice == 1:
                instrument_final = 0  # Piano
                break
            elif choice == 2:
                instrument_final = 25  # Guitar
                break
            elif choice == 3:
                instrument_final = 33  # Bass
                break
            elif choice == 4:
                instrument_final = 10  # Drums
                break
            elif choice == 5:
                instrument_final = 0  # Other
                break
            else:
                print("Invalid choice. Please enter a number between 1 and 5.")
        except ValueError:
            print("Please enter a valid number.")
    
    return instrument_to_be_changed, instrument_final #returning program numbers!


if __name__ == "__main__":    
    
    tempo = get_tempo("trial_ensemble.mp3")
    if isinstance(tempo, np.ndarray): #for older python versions
        tempo = float(tempo[0])  
    tempo = round(tempo)
    
    mp3_path = "trial_ensemble.mp3"
    song_name = os.path.splitext(os.path.basename(mp3_path))[0]  # "trial_ensemble"
    print(f"The tempo of the song {song_name} is {tempo} bpm") #flag
    # Separar instrumentos
    separate_instruments(mp3_path)
    print("Successfully separated the instruments")

    convert_mp3_to_musicxml(mp3_path, tempo, output_dir=None) 

    #modifying the songs
    input_modified_midi_path = "midi_files/trial_ensemble_piano.mid"
    output_modified_midi_path = "modified_midis/trial_ensemble_modified.mid"

    # Crear el directorio si no existe
    os.makedirs("modified_midis", exist_ok=True)

    # The next task: modify instruments within the whole song. This step is unfinished
    #original_instrument_program, new_instrument_program = choose_instrument()
    #changing_piano_to_guitar(input_modified_midi_path, output_modified_midi_path)

    midi_input_path = "midi_files/trial_ensemble_drums.mid"
    xml_output_path = "midi_files/trial_ensemble_drums1.xml"

    #to put all the midis together in a single midi file
    midi_files = glob.glob("./midi_files/*.mid") # also could use: files = os.listdir("midi_files") but glob.glob comes in handy for filtering and only taking midi files
    #glob.glob will automatically return a list of strings, each string is the (relative) path to a midi file
    print("MIDIs encontrados:")
    for midi_file in midi_files:
        print(f"- {midi_file}")

    if midi_files:  # Verifying that files were found
        output_midis_path = f"final_combined_midi/{song_name}_combined.mid"
        os.makedirs(os.path.dirname(output_midis_path), exist_ok=True)
        combined_midi = putting_midis_together(midi_files, output_midis_path)
        print(f"MIDIs combined saved 1 in: {combined_midi}")
    else:
        print("No MIDI files found to combine")

    # Create folder for final WAVs if it doesn't exist
    final_wavs_dir = "./final_combined_wav"
    os.makedirs(final_wavs_dir, exist_ok=True)

    # path for MIDI combined and final WAV
    midi_path = f"final_combined_midi/{song_name}_combined.mid"
    final_wav_output_path = os.path.join(final_wavs_dir, f"{song_name}_final.wav")
    
    # Convert MIDI to WAV
    final_wav = midi_to_wav(midi_path, final_wav_output_path)
    if final_wav:
        print(f"Conversion successful. WAV saved in: {final_wav}")
    else:
        print("Error in the conversion MIDI to WAV")
 
    
    

    