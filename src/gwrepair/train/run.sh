#!/bin/bash

python3 trainer.py --root '/Users/shreyaggarwal/Desktop/work/git' --device cpu --epochs 100 -- type glitch --outdir glitch

python3 trainer.py --root '/Users/shreyaggarwal/Desktop/work/git' --device cpu -
-epochs 100 -- type waveform --outdir waveform
