import os
import numpy as np
import soundfile as sf
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullFormatter, AutoMinorLocator
import random
from adjustText import adjust_text
import argparse

# create the "out" directory if it doesn't exist
if not os.path.exists('out'):
    os.makedirs('out')

def process_large_wav(file_path, block_size, threshold_percent):
    """
    Process a large WAV file and detect rising edges where the waveform crosses the specified threshold.
    
    :param file_path: Path to the WAV file.
    :param block_size: Number of frames to read from the file in each block.
    :param threshold_percent: Threshold as a percentage of the maximum possible amplitude to detect rising edges.
    :return: A list of trigger timestamps (in seconds), the waveform data, and the frame rate of the audio.
    """
    triggers = []  # List to store detected trigger timestamps
    
    # Read the WAV file data and frame rate
    data, frame_rate = sf.read(file_path)
    
    if len(data.shape) > 1:  # Ensure the file is single-channel
        raise ValueError("This implementation supports only single-channel WAV files.")
    
    num_frames = len(data)
    
    # Determine max amplitude based on data type (normalized float or integer)
    if np.issubdtype(data.dtype, np.floating):
        max_amplitude = 1.0  # For normalized floating-point data (-1.0 to 1.0)
    else:
        max_amplitude = np.iinfo(data.dtype).max  # For integer types, get the max amplitude value

    # Calculate the threshold value based on the percentage of the max amplitude
    threshold = max_amplitude * threshold_percent  
    print(f"Threshold set to: {threshold * 100} %")

    prev_sample = 0  # Variable to store previous sample for edge detection
    for start in tqdm(range(0, num_frames, block_size), desc="Processing WAV", unit="blocks"):
        end = min(start + block_size, num_frames)
        block_data = data[start:end]

        # Detect rising edges in the current block
        rising_edges = np.where((block_data[:-1] < threshold) & (block_data[1:] >= threshold))[0]
        
        # Convert the block indices to timestamps and append to the triggers list
        trigger_times = (start + rising_edges) / frame_rate
        triggers.extend(trigger_times)

    return triggers, data, frame_rate

def save_trigger_times(trigger_times, output_file):
    """
    Save the detected trigger timestamps to a specified file.
    
    :param trigger_times: List of trigger times in seconds.
    :param output_file: Path to the output file to save the timestamps.
    """
    with open(output_file, 'w') as f:
        for t in trigger_times:
            f.write(f"{t:.6f}\n")
    print(f"Trigger times saved to {output_file}")

def plot_waveform_with_threshold(waveform, frame_rate, threshold_percent):
    """
    Plot the waveform with a threshold line overlaid, showing the first 10 seconds of the audio.

    :param waveform: Numpy array of the audio waveform.
    :param frame_rate: Frame rate (samples per second) of the audio.
    :param threshold_percent: The threshold value as a percentage of the maximum amplitude.
    """
    time_axis = np.linspace(0, len(waveform) / frame_rate, num=len(waveform))  # Generate time axis in seconds
    
    # Limit the plot to the first 10 seconds of the audio
    max_time = 10  # In seconds
    max_samples = int(max_time * frame_rate)  # Convert max time to number of samples
    
    # Slice the waveform data and time axis to only include the first 10 seconds
    waveform_10s = waveform[:max_samples]
    time_axis_10s = time_axis[:max_samples]

    plt.figure(figsize=(24, 6))
    plt.plot(time_axis_10s, waveform_10s, label="Waveform")
    plt.axhline(y=threshold_percent, color='r', linestyle='--', label="Threshold")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"Waveform with Threshold (First {max_time}s)")
    plt.legend()
    plt.grid()
    
    # Set y-axis ticks with some padding
    y_padding = 1 * 0.2  # Padding for y-axis to make the waveform visually clearer
    y_ticks = np.arange(-1 - y_padding, 1 + y_padding, 0.1)
    plt.yticks(y_ticks)
    
    # Customize x-axis ticks for better readability
    ax = plt.gca()  # Get the current axes
    ax.xaxis.set_major_locator(MultipleLocator(0.5))  # Major ticks at 0.5s intervals
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))  # Minor ticks at 0.1s intervals
    ax.xaxis.set_minor_formatter(NullFormatter())     # Hide minor tick labels
    
    # Save the plot to a file
    plt.savefig('out/waveform-first10s.png')

def plot_trigger_envelope(trigger_times, min_region_duration, max_trigger_interval):
    """
    Plot the trigger envelope showing start and end times of regions of rising edges.
    
    :param trigger_times: List of trigger timestamps in seconds.
    :param min_region_duration: Minimum duration for a region to be plotted (in seconds).
    :param max_trigger_interval: Maximum time interval allowed between adjacent triggers to group them into one region.
    """
    regions = []  # List to store the regions of rising edges
    current_region = []  # Temporary list to build regions

    # Group triggers into regions based on the max_trigger_interval
    for t in trigger_times:
        if len(current_region) == 0:
            current_region.append(t)
        elif t - current_region[-1] < max_trigger_interval:
            current_region.append(t)
        else:
            # Add the completed region to the list and start a new region
            regions.append((current_region[0], current_region[-1]))
            current_region = [t]
    
    # Add the last region if it exists
    if len(current_region) > 0:
        regions.append((current_region[0], current_region[-1]))

    # Plot the envelope for each region
    plt.figure(figsize=(24, 6))
    text_position = 0.2  # Initialize text position for the first region
    for i, region in enumerate(regions, start=1):
        start, end = region
        if end - start < min_region_duration:
            continue  # Skip regions that are too short
        
        # Generate a random dark color for each region
        random_color = (random.uniform(0, 0.5), random.uniform(0, 0.5), random.uniform(0, 0.5))  # Darker RGB colors
        
        # Plot the region with the generated color
        plt.axvspan(start, end, color=random_color, alpha=0.6, label=f"Region {i}")
        
        # Add text annotations for the start and end times of each region
        mid_point = (start + end) / 2  # Calculate the midpoint for text positioning
        plt.text(mid_point, text_position, f"Start: {start:.3f}s\nEnd: {end:.3f}s", 
                 ha='center', va='center', color=random_color, fontsize=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

        text_position += 0.1  # Update the text position for the next region
        if text_position > 0.8:
            text_position = 0.2  # Reset text position if it exceeds the limit

    # Customize the plot appearance
    plt.xlabel("Time (s)")
    plt.ylabel("Regions")
    plt.title("Trigger Envelope")
    plt.grid()
    plt.xlim(0, max(trigger_times) + 10)  # Extend x-axis to show some margin after the last trigger
    plt.savefig('out/trigger_envelope.pdf')
    plt.savefig('out/trigger_envelope.png')
    plt.show()

def main():
    # Initialize argument parser to handle command-line arguments
    parser = argparse.ArgumentParser(description="Process a WAV file and detect rising edges based on a threshold.")
    parser.add_argument('--input', type=str, default='res/test-16bit.wav', help="Path to the input WAV file")
    parser.add_argument('--output', type=str, default='out/trigger_times.txt', help="Path to the output file to save trigger times")
    parser.add_argument('--block_size', type=int, default=1024, help="Number of frames to read per block (default: 1024)")
    parser.add_argument('--thr', type=float, default=3/4, help="Threshold as a percentage of max amplitude (default: 3/4)")
    parser.add_argument('--min_region_duration', type=int, default=20, help="Minimum duration for a region to be plotted (in seconds, default: 20)")
    parser.add_argument('--max_trigger_interval', type=int, default=3, help="Maximum duration between two adjacent triggers")

    # Parse arguments from the command line
    args = parser.parse_args()

    # Process the WAV file to detect rising edges
    trigger_times, waveform, frame_rate = process_large_wav(args.input, args.block_size, args.thr)

    # Plot the waveform with the threshold line
    plot_waveform_with_threshold(waveform, frame_rate, args.thr)

    # Plot the trigger envelope with regions of rising edges
    plot_trigger_envelope(trigger_times, args.min_region_duration, args.max_trigger_interval)

    # Save the trigger timestamps to the specified output file
    if args.output:
        save_trigger_times(trigger_times, args.output)

if __name__ == "__main__":
    main()
