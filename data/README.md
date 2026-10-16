# Data

The EEG is not in this repository. The DREAMS databases are licensed CC BY-NC-ND 3.0, which does
not allow redistributing modified copies.

Source: Devuyst S., *The DREAMS Databases and Assessment Algorithm*, Zenodo,
https://zenodo.org/records/2650142 (doi:10.5281/zenodo.2650142).

The loader (`spindles/io.py`) expects `excerpt1.edf` ... `excerpt8.edf` in this folder. Each file has:

- channel 0: EEG C3-A2 in microvolts;
- channel 1: the expert 1 spindle mask (0/1, sampled at 10 Hz);
- channel 2: the expert 2 mask, for excerpts 1-6 only.

The course provided the excerpts in this form, with the files named `excerptN.prep.edf`. If you
start from the Zenodo download instead, the expert scorings come as text files of spindle
start/duration pairs. Write them into mask channels, or adapt `load_excerpt`.

Scoring windows: expert 1 annotated only the first 1000 s of each excerpt. Expert 2 annotated
all 1800 s.
