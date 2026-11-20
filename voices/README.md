# Voice Reference Assets

IOGHAL clones a target voice using OpenBMB VoxCPM2. Third-party audio clips are excluded from this repository.

## Setup

1. Record or supply a clean 48 kHz mono WAV clip (5 to 12 seconds, minimal background noise).
2. Save the clip as `voices/hal9000_ref.wav` or set `HAL_REF_WAV=/path/to/voice.wav` (or pass `--ref-wav`).
3. Optional: place an exact transcript of the clip in a sibling `.txt` file (`voices/hal9000_ref.txt`) to enable VoxCPM2 Ultimate Cloning (reference plus continuation mode). Without a sibling `.txt` file, IOGHAL uses reference-only cloning automatically.
