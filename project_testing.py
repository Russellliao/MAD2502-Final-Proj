import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, ifft
import soundfile as sf


#1. Read audio file
"""
    Reads an audio file and checks whether there is stereo or mono

    Params:
        filename (str): Path to the audio file.

    Return:
        tuple: sample_rate, left_channel, right_channel or None
        """
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

# 4. Main Pitch detection
    peak_idx = np.argmax(spectrum)
    peak_freq = xf[peak_idx]
    note = freq_to_note(peak_freq)
    print(f"Dominant Frequency: {peak_freq:.2f} Hz → Note: {note}")


"""
    Computes the Fast Fourier Transformation of an audio signal.

    Params:
        signal (array): Audio signal data.
        sample_rate (int): Sampling rate of the audio signal.

    Return:
        tuple: xf, yf, specturm, N
"""
def compute_fft(signal, sample_rate): #transform the signals which are audio amplitudes into frequencies
    N = len(signal) #The higher N, the more detailed signal
    T = 1.0 / sample_rate #time interval between samples
    yf = fft(signal) #fast fourier transform, yf means how much each frequency has in the signal
    #xf = np.fft.fftfreq(N, T)[:N//2]
    xf = np.fft.fftfreq(N, T)[:] #frequency values in Hz
    #spectrum = np.abs(yf[:N//2])
    spectrum = yf #amplitude of each frequency
    return xf, yf, spectrum, N

"""
    Converts a frequency in Hz to the nearest musical note.

    Params:
        freq (float): Frequency in Hertz.

    Return:
        str: The musical note corresponding to the frequency
    """
def freq_to_note(freq):
    if freq <= 0:
        return "frequency less than 0"
    A4 = 440.0 #reference pitch
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'] #12 semitones in an octave
    n = round(12 * np.abs(np.log2(freq / A4))) #how many semitones is the frequency away from A4
    note_index = (n + 9) % 12
    octave = 4 + ((n + 9) // 12) #octave number
    return f"{notes[note_index]}{octave}"


def apply_filter(yf, sample_rate, N, low_pass=None, high_pass=None):
    """
    Applies a low-pass or high-pass filters to the audio signal to eliminate overly high or overly low frequencies. 
    The user can set the low_pass and high_pass values to output the audio in the way they want.

    Params:
        yf (ndarray): FFT-transformed signal.
        sample_rate (int): Sampling rate of the audio.
        N (int): The number of samples.
        (optional) low_pass (float): Low-pass filter cutoff frequency.
        (optional) high_pass (float): High-pass filter cutoff frequency.

    Return:
        ndarray: The filtered FFT result.
    """
    yf_filtered = yf.copy() # A copy is made so that the original audio is not changed
    freqs = np.abs(np.fft.fftfreq(N, 1.0 / sample_rate)) # Takes the absolute value of the frequencies 

    if low_pass is not None: 
        print(f"Applying Low-Pass Filter @ {low_pass} Hz")
        yf_filtered[freqs > low_pass] = 0 # Any value above the low pass will be set to 0

    if high_pass is not None:
        print(f"Applying High-Pass Filter @ {high_pass} Hz")
        yf_filtered[freqs < high_pass] = 0 # Any frequency below the high pass is also set to 0

    filtered_signal = np.real(ifft(yf_filtered)) # Converts the signals back using inverse fft and takes only the real parts
    return yf_filtered,filtered_signal


# 6. Visualization
def plot_spectrum(xf, spectrum, note=None, label='Channel'):
    """
    Plots the frequency spectrum of the audio

    Params:
        xf (ndarray): Frequency bins.
        spectrum (ndarray): Magnitude of FFT results.
        note (str, optional): Dominant musical note to display in title.
        label (str): Label for the plot legend.
    """
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

    filtered_yf, filtered_signal = apply_filter(yf, sample_rate, N, low_pass=20000, high_pass=300)
    filtered_spectrum=np.abs(filtered_yf)
    # Plot the spectrum
    plot_spectrum(xf, spectrum, note=dominant_note, label=file)
    plot_spectrum(xf, filtered_spectrum, note=dominant_note, label=file)



