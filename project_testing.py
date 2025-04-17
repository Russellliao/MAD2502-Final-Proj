import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, ifft
import soundfile as sf


# data, sample_rate = sf.read('rock-and-roll-01-325493.mp3')
# print('File loaded successfully')
# print('Shape:', data.shape)
# print('Sample rate:', sample_rate)
    #except sf.SoundFileError as e:
         #print(f"Error reading the file: {e}")
    #except RuntimeError as re:
        #print(f"Another error occurred: {re}")


# 1. Read audio file and check if there is stereo
def read_audio(filename):
    data, sample_rate = sf.read(filename)
    print(f"Sample rate: {sample_rate} Hz")
    if data.ndim == 2:
        print("Stereo audio detected.")
        left, right = data[:, 0], data[:, 1]
    else:
        print("Mono audio detected.")
        left, right = data, None
    return sample_rate, left, right

# 2. Compute FFT and frequency bins
def compute_fft(signal, sample_rate):
    N = len(signal)
    T = 1.0 / sample_rate
    yf = fft(signal)
    xf = np.fft.fftfreq(N, T)[:N//2]
    spectrum = np.abs(yf[:N//2])
    return xf, yf, spectrum, N

# 3. Convert frequency to musical note
def freq_to_note(freq):
    if freq <= 0:
        return "N/A"
    A4 = 440.0
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    n = round(12 * np.log2(freq / A4))
    note_index = (n + 9) % 12
    octave = 4 + ((n + 9) // 12)
    return f"{notes[note_index]}{octave}"

# 4. Filters
def apply_filter(yf, sample_rate, N, low_pass=None, high_pass=None):
    yf_filtered = yf.copy()
    freqs = np.abs(np.fft.fftfreq(N, 1.0 / sample_rate))

    if low_pass:
        print(f"Applying Low-Pass Filter @ {low_pass} Hz")
        yf_filtered[freqs > low_pass] = 0

    if high_pass:
        print(f"Applying High-Pass Filter @ {high_pass} Hz")
        yf_filtered[freqs < high_pass] = 0

    return yf_filtered

# 5. Visualization
def plot_spectrum(xf, spectrum, note=None, label='Channel'):
    plt.plot(xf, spectrum, label=label)
    if note:
        plt.title(f"Frequency Spectrum (Dominant Note: {note})")
    else:
        plt.title("Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

sample_rate, left, right = read_audio('rock-and-roll-01-325493.mp3')


