Terrain policy data (public data only; nothing game-derived).

dem_india.bin, dem_america.bin
  LOSSLESS crops of NOAA NCEI ETOPO 2022 60 arc-second Global Relief Model, surface elevation (z), doi:10.25921/fd45-gt74 (public domain),
  https://www.ngdc.noaa.gov/thredds/ncss/grid/global/ETOPO2022/60s/60s_surface_elev_netcdf/ETOPO_2022_v1_60s_N90W180_surface.nc
  cropped to the fixed India (theatre 3) / America (theatre 1) canvas footprints + 40 cell margin. Every float32 cell is stored bit-exactly
  (NaN -> 0 as dem.get_dem/geo.Dem do): 1/16 m lattice integer + float32-ulp offset, MED-predicted, context-split zlib.
  Layout and transform: tools/terrain_paint/prepare_dem.py. Decoded by src/terrain/art/dem.rs (include_bytes!, SHA-256 trailer verified at load).
  Total 75753474 bytes for both grids. Verified bit-for-bit (0 differing cells in both grids) by decoding the embedded files with dem.rs in a harness and
  comparing every float32 bit pattern with the NetCDF-derived grid whose SHA-256 is given below as grid_float32_sha256.

calibration.json
  World <-> lon/lat affines of theatres 1 and 3 (least-squares fits of public gazetteer place coordinates to the theatre map coordinates;
  the offline fitter and its anchor exports are not shipped) and the 41-point pooled metres -> height byte quantile map (a numeric curve that
  was matched once against the ETW height-byte distribution; it contains no heightmap pixels). Authored numeric constants.

Regenerate (dev only): python -B tools/terrain_paint/prepare_dem.py [--cache DIR]   (NetCDF tiles are fetched from NOAA when absent)

Measured (prepare_dem.py output; grid_float32_sha256 = SHA-256 of the little-endian float32 row-major NetCDF-derived grid):
{
 "india": {
  "file": "dem_india.bin",
  "bytes": 28399367,
  "layer1_bytes": 18522026,
  "layer2_bytes": 9874783,
  "cols": 3647,
  "rows": 3996,
  "ci0": 14291,
  "rj0": 4336,
  "lon_range": [
   58.1917,
   118.9583
  ],
  "lat_range": [
   -17.725,
   48.8583
  ],
  "land_cells": 7124611,
  "sea_cells": 7448712,
  "sha256": "19432c5a10ba5ecd8d9587ea64bf47139f973053439a64068d799621c54afbc6",
  "grid_float32_sha256": "e5f5ed6b170b2d0a5974343229a9dfd04f0b9cc27b78dd66a6f1c69542643761"
 },
 "america": {
  "file": "dem_america.bin",
  "bytes": 47354107,
  "layer1_bytes": 28819844,
  "layer2_bytes": 18531677,
  "cols": 5160,
  "rows": 4465,
  "ci0": 2986,
  "rj0": 5024,
  "lon_range": [
   -130.225,
   -44.2417
  ],
  "lat_range": [
   -6.2583,
   68.1417
  ],
  "land_cells": 10067478,
  "sea_cells": 12969095,
  "sha256": "1ea96c54b0a6ed2b10c8fe3f2f407483da8de784bafe0c3fce84ce66e99278cd",
  "grid_float32_sha256": "670004d98ffc432037c4c0c8254adda48ad664f23c09a99be504eb0d73b56229"
 }
}
