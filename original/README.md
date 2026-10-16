# 2023 course code (unchanged)

These are the original files from the October 2023 team project (FH Technikum Wien, Biosignal
Processing, Sprint 1-2). They are kept exactly as submitted, so the before/after comparison in the
main README can be checked.

`spindle_detection_2023.py` reads `trainingsData/excerptN.prep.edf` relative to the working
directory. Run unchanged, it prints a test-split F1 of 25.3% (scorer 1) and 28.1% (scorer 2).

The same detector is ported into `spindles/detect.py::detect_morlet_2023`. The detection logic is
unchanged; only the output is converted to seconds. That port is what gets scored correctly in
the main results.
