# Task: Crosstalk and baseline-noise analysis for a 16-channel biosignal acquisition system

You are a research and engineering assistant. Write a Python analysis pipeline (numpy, scipy, pandas, matplotlib) for two benchtop tests of my 16-channel ADS1299-based EMG/EEG acquisition system. The pipeline must be reproducible, well commented, and produce publication-quality figures and tables for a thesis.

## Paths (I fill these in manually)
- Baseline recordings (breakout PCB, all channels shorted): BASELINE_PATH = "/home/karl/Documents/Master_Thesis/Testing/benchtop_testing/breakout_pcb_baseline"
- Crosstalk recordings (one file per driven channel / frequency / amplitude): CROSSTALK_PATH = "/home/karl/Documents/Master_Thesis/Testing/benchtop_testing/crosstalk_test"
- Output directory for figures and tables: OUTPUT_DIR = "/home/karl/repos/Sokosti_tools/benchtop_testing_tools/crosstalk_test"
- Optional layout / channel-to-ADC mapping file: LAYOUT_PATH = "/home/karl/repos/Sokosti_tools/benchtop_testing_tools/crosstalk_test"

Put all of these at the top of the script as constants. Do not hard-code anything else about file locations.

## System parameters (verify and ask me if unclear, do not guess silently)
- Sampling rate fs: 1000 Hz
- PGA gain: 1, shown by the 1G in the folder name
- Vref: 4.5 V (derive LSB = 2*Vref / (gain * 2^24), and state the formula used)
- Channel count: 16 (channel-to-ADS1299 chip mapping: channel 1-8 are ads1299-1 and channel 9-16 are ads1299-2 )
- File format and column layout of the PC recordings:  
    sample,status1_ok,status2_ok,ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8,ch9,ch10,ch11,ch12,ch13,ch14,ch15,ch16,roll,pitch,yaw,accel_x,accel_y,accel_z
    80992,1,1,484,-897,-911,-1006,-919,-896,-929,-972,-767,-903,-912,-893,-769,-901,-881,-896,,,,,,
    80993,1,1,364,-899,-908,-1012,-914,-896,-926,-964,-770,-901,-916,-894,-766,-902,-883,-896,,,,,,
    80994,1,1,243,-899,-903,-1014,-914,-893,-923,-969,-769,-897,-922,-888,-761,-905,-885,-894,,,,,,
    80995,1,1,125,-899,-904,-1007,-919,-899,-925,-971,-769,-899,-921,-891,-756,-906,-884,-894,,,,,,
    
    only emg channels populated, imu is off.
    
    (describe, or I will paste the first lines)
- Filename convention that encodes driven channel, frequency, and amplitude: CH1_1G_Sine_2mVpp_100Hz_PC.csv - only use PC.csv endings. CH1: channel 1, 1G: gain 1, Sine: sine waveform, 2mVpp: amplitude, 100Hz: frequency, PC: recorded on PC and not on SD card.

## Data source
- Use only the PC (BLE-streamed) recordings. Do NOT use the SD card recordings, because they have more lost samples.
- BLE can also lose packets. Detect gaps via the sample counter / timestamp, report the loss per file, and flag any file with significant loss. Do not silently interpolate. If gaps exist, either analyse only the longest gap-free section or state clearly how they were handled.

## Test descriptions
**Baseline test:** Patient_Reference and Bias-Out are connected together, and all channels are shorted to Patient_Reference. All channels are recorded. Purpose: noise floor, used for SNR and for the crosstalk detection limit.

**Crosstalk test:** Patient_Reference and Bias-Out are connected together. One channel at a time is driven from a signal generator (via a BNC adapter PCB), and all other channels stay tied to Patient_Reference (near-zero source impedance). All 16 channels are recorded. Conditions: sine at 10 Hz and 100 Hz, at 2 mVpp and 10 mVpp. Recording length is about 15 s per run (check the actual length from the data).

## Processing requirements
1. Convert ADC counts to volts (µV for display) using LSB and gain. Remove the mean (DC) per analysis segment.
2. Use the whole recording from the first sample. The signal had already settled before recording started, so do NOT discard any initial data.
3. Split each recording into equal segments of 3 s (about 5 segments, make the segment length a parameter) to get mean ± SD of every metric across segments.
4. Amplitude extraction at the injected frequency, using the identical method for the driven channel, the victim channels, and the baseline:
   - Primary: least-squares sine fit. The generator and the ADC clock are not locked, so fit the frequency freely (4-parameter fit, with the nominal frequency as starting value and a tight bound).
   - Cross-check: Hann-windowed FFT with amplitude A = 2*|X[k]| / sum(w).
   - Sanity check on the driven channel: the sine-fit amplitude, the FFT amplitude, and RMS*sqrt(2) must agree (report the percent difference). Also compare with the generator setting.
   - Do not use broadband RMS for victim channels.
5. Noise reference from the baseline: use the same method and segment length to get the noise amplitude in the same narrow band (or bin) around 10 Hz and 100 Hz, per channel. Also compute band-limited RMS noise for 0.5-100 Hz and 20-450 Hz, and noise density in nV/sqrt(Hz) (Welch, scaling='density'). Use scaling='spectrum' only for reading tone amplitudes.
6. Crosstalk metric per (driven channel, victim channel, frequency, amplitude):
   XT_dB = 20*log10(A_victim / A_driven)
   - If A_victim is at least 6 dB above the baseline noise amplitude at that frequency: noise-correct with A_ct = sqrt(A_meas^2 - A_noise^2).
   - Otherwise, mark the value as censored and report it as an upper bound ("< X dB", at the noise-floor level). Never plot censored values as real crosstalk. Show them visibly distinct (hatching or open markers).
   - Detection limit per condition: 20*log10(A_noise / A_driven).

## Figures (matplotlib, consistent style, readable at thesis column width, axis labels with units, vector PDF + PNG output)
Crosstalk test:
1. Setup and parameter overview (a table, plus a simple block diagram if feasible).
2. PSD overlay: driven channel, one or two victim channels (e.g. an adjacent and a far one), and a baseline channel, with the test frequency marked.
3. Crosstalk matrix heatmap (rows = driven channel, columns = victim channel, color = dB, diagonal masked, censored cells hatched). One panel per frequency/amplitude combination (4 panels), shared color scale.
4. Crosstalk vs. channel distance: scatter of all pairs, with mean and max per distance, colored by frequency, plus a horizontal detection-limit line. Distinguish adjacent, same-ADS1299, and other-ADS1299 pairs. If LAYOUT_PATH provides physical coordinates, also plot against physical distance.
5. Frequency and amplitude dependence: compare 10 Hz vs 100 Hz (is there a ~20 dB/decade rise, which would suggest capacitive coupling?) and 2 mVpp vs 10 mVpp (linearity). Quantify the differences.
6. Summary table per frequency/amplitude: worst-case pair and its dB value, median over all pairs, mean for adjacent vs non-adjacent, number of censored pairs, detection limit in dB. Export as CSV and as LaTeX (booktabs).

Baseline test:
7. Per-channel noise bar chart (µV RMS in 0.5-100 Hz and 20-450 Hz), with ADS1299 chip boundaries marked.
8. PSD overlay of all 16 channels in nV/sqrt(Hz), with 50 Hz and its harmonics marked if present.
9. Table per channel: mean, RMS, Vpp, band-limited noise, and the median across channels (CSV + LaTeX).
10. SNR helper: SNR = 20*log10(A_sig / noise_RMS_in_band), with the band definition stated explicitly, so I can reuse it for the known-signal test.

## Caveats to state in the output (as a short text block)
- Victim inputs are tied to Patient_Reference (near-zero source impedance); real electrodes have kOhm-level impedance, so real-world coupling will likely be worse.
- Only two test frequencies, so no full frequency response can be claimed.
- Generator output and BNC adapter PCB layout may add coupling that does not originate from the analog front-end.
- Short recordings (~15 s) limit the statistical power (few segments).

## Code style
- I strongly prefer minimal, targeted code. Keep it modular: small functions (load, convert, extract_amplitude, noise_reference, crosstalk_matrix, plotting functions), no unnecessary abstractions. If I later ask for changes, edit the relevant function only, do not rewrite the whole script.
- Print a short log: files processed, sample counts, packet loss, any channels or files that failed sanity checks.
- If any required information is missing (file format, gain, channel-to-chip mapping), ask me before writing code instead of assuming.