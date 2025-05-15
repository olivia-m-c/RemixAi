import numpy as np
import librosa
import pretty_midi
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = fs * 0.5  # Frecuencia de Nyquist
    low = lowcut / nyq 
    high = highcut / nyq
    
    # Asegurarnos de que los valores estén entre 0 y 1
    low = max(0.001, min(0.99, low))
    high = max(0.001, min(0.99, high))
    
    print(f"Low: {low}, High: {high}")  # Debug
    
    b, a = butter(order, [low, high], btype='band') #only lets through frequencies between low and high
    #order is how steep the filter is, higher = steeper
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def wav_to_drum_midi(wav_file, output_file):
    """
    Convert drum hits in a WAV file to MIDI events.
    
    Parameters:
    wav_file (str): Path to input WAV file
    output_file (str): Path to output MIDI file
    """
    # Cargar el archivo WAV
    y, sr = librosa.load(wav_file)
    
    # Limpieza exhaustiva de la señal
    y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
    y = np.clip(y, -1.0, 1.0)
    
    # Eliminar cualquier valor no finito
    mask = np.isfinite(y)
    if not np.all(mask):
        print("Encontrados valores no finitos, limpiando...")
        y = y[mask]
    
    # Normalizar y verificar
    y = librosa.util.normalize(y)
    
    # Verificación estricta
    if not np.all(np.isfinite(y)):
        raise ValueError("No se pudo limpiar la señal correctamente")
    
    # Reducir el orden del filtro Butterworth
    y_hihat = bandpass_filter(y, 1200, 16000, sr, order=3)  # adjusted for eighth hihats
    y_snare = bandpass_filter(y, 200, 700, sr, order=3)
    y_kick = bandpass_filter(y, 30, 150, sr, order=3)
    
    # Usar parámetros más seguros para onset_detect
    onset_hihat = librosa.onset.onset_detect(
        y=y_hihat, sr=sr,
        delta=0.03,
        wait=2,
        pre_max=20,
        post_max=20,
        pre_avg=100,
        post_avg=100,
        units='time'
    )
    
    # Ajustar detección para kick y snare
    onset_kick = librosa.onset.onset_detect(
        y=y_kick, sr=sr,
        delta=0.2,     # Menos sensible para evitar falsos positivos
        wait=10,       # Mayor espera para evitar detecciones múltiples
        units='time'
    )
    
    onset_snare = librosa.onset.onset_detect(
        y=y_snare, sr=sr,
        delta=0.2,
        wait=10,
        units='time'
    )
    
    # Filter signals for different drums using specific frequency ranges
    y_high_tom = bandpass_filter(y, 450, 550, sr)  # High tom
    #y_mid_tom = bandpass_filter(y, 350, 450, sr)  # Mid tom
    y_low_tom = bandpass_filter(y, 200, 350, sr)  # Low tom
    y_crash = bandpass_filter(y, 5000, 15000, sr)  # Crash cymbal
    y_ride = bandpass_filter(y, 2000, 7000, sr)  # Ride cymbal
    
    # Calculate RMS energy for cymbal decay detection
    def get_cymbal_envelope(signal, frame_length=2048):
        return librosa.feature.rms(y=signal, frame_length=frame_length)[0]
    
    crash_env = get_cymbal_envelope(y_crash)
    ride_env = get_cymbal_envelope(y_ride)
    
    # Ajustar los umbrales de peaks
    hihat_peaks, _ = find_peaks(np.abs(y_hihat), height=0.01, distance=int(sr/8))  # determinant for eighth hihats
    kick_peaks, _ = find_peaks(np.abs(y_kick), height=0.15, distance=int(sr/4))
    snare_peaks, _ = find_peaks(np.abs(y_snare), height=0.15, distance=int(sr/4))
    high_tom_peaks, _ = find_peaks(np.abs(y_high_tom), height=0.1, distance=50)
    #mid_tom_peaks, _ = find_peaks(np.abs(y_mid_tom), height=0.1, distance=50)
    #low_tom_peaks, _ = find_peaks(np.abs(y_low_tom), height=0.1, distance=50)
    crash_peaks, _ = find_peaks(crash_env, height=0.1, distance=50)
    ride_peaks, _ = find_peaks(ride_env, height=0.1, distance=50)
    
    # Create a MIDI file
    pm = pretty_midi.PrettyMIDI()
    drum_program = pretty_midi.Instrument(program=0, is_drum=True)
    
    # MIDI note numbers for drum kit (General MIDI standard)
    KICK = 36      # Bass Drum 1
    SNARE = 38     # Acoustic Snare
    HIHAT = 42     # Closed Hi-Hat
    HIGH_TOM = 50  # High Tom
    MID_TOM = 47   # Mid Tom
    LOW_TOM = 45   # Low Tom
    CRASH = 49     # Crash Cymbal 1
    RIDE = 51      # Ride Cymbal 1
    
    # Convert peak positions to MIDI notes with variable note duration
    def peaks_to_notes(peak_positions, note_number, velocity=100, duration=0.1):
        for peak in peak_positions:
            # Get the amplitude at the peak for velocity
            amplitude = np.abs(y[peak])
            # Scale amplitude to MIDI velocity (0-127)
            scaled_velocity = min(127, int(velocity * (amplitude / 0.1)))
            
            note = pretty_midi.Note(
                velocity=scaled_velocity,
                pitch=note_number,
                start=peak/sr,  # Convert sample position to seconds
                end=(peak/sr) + duration  # Note duration
            )
            drum_program.notes.append(note)
    
    # Add notes for each drum type
    peaks_to_notes(kick_peaks, KICK, velocity=100)
    peaks_to_notes(snare_peaks, SNARE, velocity=90)
    peaks_to_notes(hihat_peaks, HIHAT, velocity=70) 
    peaks_to_notes(high_tom_peaks, HIGH_TOM, velocity=85)
    #peaks_to_notes(mid_tom_peaks, MID_TOM, velocity=85)
    #peaks_to_notes(low_tom_peaks, LOW_TOM, velocity=85)
    # Longer duration for cymbals to capture their decay
    peaks_to_notes(crash_peaks, CRASH, velocity=90, duration=1.0)
    peaks_to_notes(ride_peaks, RIDE, velocity=85, duration=0.5)
    
    # Add the drum program to the MIDI file
    pm.instruments.append(drum_program)
    
    # Save the MIDI file
    pm.write(output_file)
    
    drum_counts = {
        'kick_hits': len(kick_peaks),
        'snare_hits': len(snare_peaks),
        'hihat_hits': len(hihat_peaks),
        'high_tom_hits': len(high_tom_peaks),
        #'mid_tom_hits': len(mid_tom_peaks),
        #'low_tom_hits': len(low_tom_peaks),
        'crash_hits': len(crash_peaks),
        'ride_hits': len(ride_peaks)
    }
    
    return drum_counts

def wav_to_numpy(wav_file):
    # Cargar el archivo WAV
    y, sr = librosa.load(wav_file)
    
    # Convertir a array numpy y normalizar
    audio_array = np.array(y)
    
    # Analizar frecuencias
    D = librosa.stft(audio_array)
    frequencies = librosa.fft_frequencies(sr=sr)
    magnitudes = np.abs(D)
    
    # Encontrar las frecuencias dominantes
    peak_frequencies = frequencies[np.argmax(magnitudes, axis=0)]
    print("\nFrecuencias dominantes:", peak_frequencies[:20])  # Mostrar primeras 20
    
    # Visualizar el espectrograma
    plt.figure(figsize=(12, 8))
    D_db = librosa.amplitude_to_db(magnitudes, ref=np.max)
    librosa.display.specshow(D_db, y_axis='log', x_axis='time', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Espectrograma con frecuencias')
    plt.show()
    
    # Guardar array
    np.save('drums_array.npy', audio_array)
    f=open('drums_array.txt', 'w')
    f.write(str(audio_array))
    f.close()
    
    return audio_array, sr

if __name__ == "__main__":
    wav_file = "separated_wavs/trial_ensemble_drums.wav"
    output_file = "separated_wavs/trial_ensemble_drums4.mid"
    drum_counts = wav_to_drum_midi(wav_file, output_file)
    print(drum_counts)

    # Probar la función
    audio_array, sr = wav_to_numpy(wav_file)
    print("Shape del array:", audio_array.shape)
    print("Frecuencia de muestreo:", sr)