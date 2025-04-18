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
def read_audio(filename): #Import the audio file into python and compare the data in left and right channel; if they are different, there is stereo audio
    data, sample_rate = sf.read(filename)
    print(f"Sample rate: {sample_rate} Hz") #usually 44.1kHz or 48kHz
    if data.ndim == 2: 
        print("Stereo audio detected.")
        left, right = data[:, 0], data[:, 1]
    else:
        print("Mono audio detected.")
        left, right = data, None
    return sample_rate, left, right

# 2. Fourier Transformation
def compute_fft(signal, sample_rate): #transform the signals which are audio amplitudes into frequencies
    N = len(signal) #The higher N, the more detailed signal
    T = 1.0 / sample_rate #time interval between samples
    yf = fft(signal) #fast fourier transform, yf means how much each frequency has in the signal
    #xf = np.fft.fftfreq(N, T)[:N//2]
    xf = np.fft.fftfreq(N, T)[:] #frequency values in Hz
    #spectrum = np.abs(yf[:N//2])
    spectrum = np.abs(yf[:]) #amplitude of each frequency
    return xf, yf, spectrum, N

# 3. Convert frequency to musical note
def freq_to_note(freq):
    if freq <= 0:
        return "N/A"
    A4 = 440.0 #reference pitch
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'] #12 semitones in an octave
    n = round(12 * np.log2(freq / A4)) #how many semitones is the frequency away from A4
    note_index = (n + 9) % 12
    octave = 4 + ((n + 9) // 12) #octave number
    return f"{notes[note_index]}{octave}"

# 4. Main Pitch detection
peak_idx = np.argmax(spectrum)
peak_freq = xf[peak_idx]
note = freq_to_note(peak_freq)
print(f"Dominant Frequency: {peak_freq:.2f} Hz → Note: {note}")

# 5. Filters
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

# 6. Visualization
def plot_spectrum(xf, spectrum, note=None, label='Channel'):
    plt.plot(xf, spectrum, label=label)
    if note:
        plt.title(f"Frequency Spectrum (Dominant Note: {note})")
    else:
        plt.title("Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.xlim(0, 500)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


#sample_rate, left, right = read_audio('rock-and-roll-01-325493.mp3')


filenames = [
    'rock-and-roll.mp3',
    'edm.mp3',
    'twisterion-b1-221376.mp3',
    'oasis.mp3'
]

# Loop through each file and process it
for file in filenames:
    print(f"\n--- Processing: {file} ---")
    sample_rate, left, right = read_audio(file)
    xf, yf, spectrum, N = compute_fft(left, sample_rate)

    # Get index of the dominant frequency (highest amplitude)
    dominant_idx = np.argmax(spectrum)
    dominant_freq = xf[dominant_idx]
    dominant_note = freq_to_note(dominant_freq)

    # Plot the spectrum
    plot_spectrum(xf, spectrum, note=dominant_note, label=file)



