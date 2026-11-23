# antenna-persona-5

Learn the full-sphere radiation pattern (dB GainTotal, 181 theta x 360 phi) of a 32x32
planar array (62.5 mm pitch, 2.4 GHz) from its complex excitation (drive table), for any
number of simultaneous beams.

## Model

Hybrid physics + neural network (`model.py`):

1. **Physics stage** (~3.4k parameters): learnable 7x7 complex mutual-coupling kernel and
   per-element edge gains -> exact array factor on the sphere -> learnable smooth embedded
   element pattern and floor -> normalised to directivity + learnable efficiency. Linear in
   the excitation, so 1, 2, 3 (or more) beams are handled by construction.
2. **Residual U-Net** (~7.9M parameters): operates on the (theta, phi) map, circular in phi,
   conditioned on an encoding of the weight map; predicts a dB correction to the physics
   pattern (sidelobe structure, back lobes, floor).

## Usage

```powershell
python prepare_data.py                  # CSVs -> data/dataset.npz (once)
powershell -File run_overnight.ps1      # main run + 3-beam generalisation check
python evaluate.py --run runs/main      # plots + metrics table
python predict.py --run runs/main drive_tables/nb2_0007.csv --out predictions
```

`python train.py --help` lists options (wall-clock budget `--hours`, `--train-beams`, ...).
