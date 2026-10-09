# Public elevation-data preparation

The editor now paints fixed-theatre terrain natively from the user's ETW snapshot. The old Python painters, calibration fitter, game-derived anchor exports and replacement-art packager are retired. No game artwork is generated or distributed by this developer tool.

`prepare_dem.py` reproduces the public elevation data compiled into the editor:

- `assets/policy/terrain/dem_india.bin`
- `assets/policy/terrain/dem_america.bin`
- `assets/policy/terrain/calibration.json` (copied from the authored input)
- `assets/policy/terrain/README.txt` (provenance and measured sizes)

## Inputs and dependencies

Python with **NumPy and SciPy** is required only for this preparation step, not for building or running the Rust editor. `dem/dem.py` reads cached 10-degree NetCDF tiles and downloads missing tiles from the [NOAA ETOPO 2022 60-arc-second surface service](https://www.ngdc.noaa.gov/thredds/ncss/grid/global/ETOPO2022/60s/60s_surface_elev_netcdf/ETOPO_2022_v1_60s_N90W180_surface.nc). Dataset: [ETOPO 2022, DOI 10.25921/fd45-gt74](https://doi.org/10.25921/fd45-gt74).

No game files, original artwork, extracted region lists or previously generated game resources are read. The calibration defaults to the authored `assets/policy/terrain/calibration.json`; `--calibration FILE` selects another explicit calibration. The cache directory is selected by `--cache`, then `TERRAIN_DEM_CACHE`, then `dem/cache/` beside the helper.

## Reproduce without overwriting shipped data

Run from the repository root, choosing an output and cache directory on the intended drive:

```powershell
python -B tools/terrain_paint/prepare_dem.py --cache F:\NTW2\work\dem\cache --out F:\NTW2\work\dem-repro
```

Omit `--out` only when intentionally replacing `assets/policy/terrain/`. Missing cache tiles require network access; a complete cache permits offline regeneration. Compare the generated binaries and calibration against the shipped files before replacing them. Changing these data requires rebuilding the executable and invalidates the native terrain cache.

## Encoding and verification

The grids retain **every float32 bit**, rather than quantizing elevations. The `KGCDEM02` format separates a predicted integer layer and an exact residual layer into context-dependent deflate streams. `prepare_dem.py` documents the complete layout and asserts reconstruction of the original float32 values; `src/terrain/art/dem.rs` is the native decoder. The output records a raw-grid SHA-256 and a payload checksum. Compressed byte reproducibility also depends on the zlib version.

A real run against the existing NOAA cache reproduced both shipped `.bin` files and `calibration.json` byte-for-byte after removal of the old projection/fitter dependencies. The lossless elevation files total 75,753,474 bytes. Runtime terrain generation and cache handling live in `src/terrain/art/` and `src/terrain/fixed_art.rs`; see the main README for independent old/new export comparisons and the remaining in-game verification limit.
