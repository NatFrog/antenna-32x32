# 32x32 experiment

Learn the full-sphere radiation pattern (dB GainTotal, 181 theta x 360 phi) of a 32x32
planar array (62.5 mm pitch, 2.4 GHz) from its complex excitation, for any
number of simultaneous beams (currently trained and tested with 1-3 beams)

## Model

Hybrid physics + neural network (U-Net):

1. **Physics stage**: learnable 7x7 complex mutual-coupling kernel and
   per-element edge gains -> exact array factor on the sphere -> learnable smooth embedded
   element pattern and floor -> normalised to directivity + learnable efficiency. Linear in
   the excitation, so 1, 2, 3 (or more) beams are handled by construction.
2. **Residual U-Net**: operates on the (theta, phi) map, circular in phi,
   conditioned on an encoding of the weight map; predicts a dB correction to the physics
   pattern (sidelobe structure, back lobes, floor).

## Usage

```powershell
python prepare_data.py                  # CSVs -> data/dataset.npz
python evaluate.py --run runs/main      # plots + metrics table
python predict.py --run runs/main drive_tables/nb2_0007.csv --out predictions
```
