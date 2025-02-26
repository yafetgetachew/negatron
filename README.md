# NEGATRON

A simple desktop application that converts  color negative film scans into color positive images.

The conversion is essentially a color inversion algorithm—with an added twist. It first computes the negative by subtracting each normalized pixel value from 1 (the standard inversion) and then scales each channel by a factor based on a base color (either auto-detected or manually picked). This extra normalization adjusts the contrast according to the base color.

## Installation

1. Clone or download this repository.
2. Install the required packages with:

   ```bash
   pip install -r requirements.txt

3. Run the app

   ```bash
   python main.py
