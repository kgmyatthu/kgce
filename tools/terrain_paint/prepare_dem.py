"""Dev-time only: build the LOSSLESS public-DEM data shipped under assets/policy/terrain/ from NOAA ETOPO 2022 (60 arc-second surface).

    python -B prepare_dem.py [--cache DIR] [--out DIR] [--calibration FILE]

Reads the 10x10 degree NetCDF tiles of dem/dem.py (cache dir = --cache | $TERRAIN_DEM_CACHE | ./dem/cache; missing tiles are fetched from the
NOAA NCSS service named in dem/dem.py), crops each native theatre footprint (NaN -> 0) and writes
    dem_india.bin  dem_america.bin   bit-exact float32 grids, MED-predicted, context-split deflate (format below; decoded by src/terrain/art/dem.rs)
    calibration.json                 copy of the authored affine fits and height curve supplied with --calibration
    README.txt                       provenance + measured sizes
The output is a pure function of the NetCDF tiles + calibration.json + this file (+ the zlib build used for level 9), so
re-running it on the same inputs reproduces the same bytes. No game files are read or written; Python is not needed at run time.
The encoder asserts that the transform is exactly invertible (every float32 bit pattern reconstructed).

Footprint = bounding box of the theatre's level-0 canvas corners mapped through the theatre's affine inverse (America canvas world x -920..-320,
y -40..480; India canvas x 420..840, y -120..340), grown by MARGIN cells (spline prefilter decays by 0.268 per cell, jitter is <= ~5 cells).
Cell centres: lon = (col + 0.5)/60 - 180, lat = (row + 0.5)/60 - 90, rows ascending northwards.

Lossless transform of every float32 value v (no NaN/inf/-0 allowed; asserted):
    q = rint(16 v)                      integer lattice layer (ETOPO sea values are almost all exact multiples of 1/16 m)
    w = float32(q / 16)                 exact (|q| < 2^24)
    d = ord(v) - ord(w)                 ord(x) = order-preserving signed integer of the IEEE-754 bits (sign-magnitude -> two's complement), i.e. the
                                        distance in float32 ulps (0 for lattice values; the land fractions are full-mantissa floats, d is their noise)
Layer 1: r = q - MED(W, N, NW) (out of grid = 0, row 0 = south), zigzag u32, contexts ctx = digitize(|r_W| + |r_N|, CTX_BINS).
Layer 2: zigzag(d) u32, groups g = ((bitlength(|q|) * 2 + (q < 0)) * 2 + (d_W != 0)) * 2 + (d_N != 0), 24 bit-length classes -> 192 groups.
Every context/group stream (cells in raster order) is split into little-endian byte planes (only as many as its maximum needs) and each plane is
zlib level 9 compressed.

File layout (little endian):
    magic "KGCDEM02" | u32 ncols | u32 nrows | i32 ci0 | i32 rj0 | u32 cells_per_degree(60)
    u32 nbins | i32 bins[nbins]                    (nctx = nbins + 1)
    nctx   * { u32 count, u8 nplanes, u32 comp_len[nplanes] }     layer-1 contexts
    192    * { u32 count, u8 nplanes, u32 comp_len[nplanes] }     layer-2 groups
    zlib streams in the same order (layer 1 then layer 2, context-major, plane 0 = low byte) | 32-byte SHA-256 of everything before it
ci0/rj0 are global cell indices of the first stored column / (south) row on the 1/60 degree ETOPO lattice.
"""
import argparse, hashlib, json, math, os, struct, sys, zlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEM_DIR = os.path.join(HERE, 'dem')
POLICY_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'assets', 'policy', 'terrain')

MARGIN = 40
CANVAS = {'3': (420.0, 840.0, -120.0, 340.0), '1': (-920.0, -320.0, -40.0, 480.0)}   # world x0, x1, y0(south), y1(north)
NAMES = {'3': 'india', '1': 'america'}
LAT_SCALE = 16.0
CTX_BINS = [1, 3, 6, 12, 24, 48, 120, 300, 800]
NBL = 24
NGROUPS = NBL * 8
MAGIC = b'KGCDEM02'



def ordered(x):
    """float32 -> int32 monotone in value (sign-magnitude IEEE bits -> two's complement)."""
    u = x.view(np.uint32)
    mag = (u & np.uint32(0x7fffffff)).astype(np.int32)
    return np.where(u >> np.uint32(31) == 1, -mag, mag)


def unordered(o):
    mag = np.abs(o.astype(np.int64)).astype(np.uint32)
    return np.where(o < 0, mag | np.uint32(0x80000000), mag).astype(np.uint32).view(np.float32)


def shift(a, dy, dx):
    """out[r, c] = a[r - dy, c - dx] with zeros outside (dy, dx in {0, 1})."""
    o = np.zeros_like(a)
    o[dy:, dx:] = a[:a.shape[0] - dy, :a.shape[1] - dx]
    return o


def med(q):
    W = shift(q, 0, 1); N = shift(q, 1, 0); NW = shift(q, 1, 1)
    mx = np.maximum(W, N); mn = np.minimum(W, N)
    return np.where(NW >= mx, mn, np.where(NW <= mn, mx, W + N - NW))


def zigzag(r):
    r = r.astype(np.int64)
    return ((r << 1) ^ (r >> 63)).astype(np.uint32)


def streams(zz, grp, ngroups):
    """-> (table bytes, body bytes) of the per-group byte-plane zlib streams (raster order inside a group)."""
    order = np.argsort(grp.ravel(), kind='stable')
    zs = zz.ravel()[order]
    counts = np.bincount(grp.ravel(), minlength=ngroups)
    del order
    table = b''; body = b''; pos = 0
    for g in range(ngroups):
        v = zs[pos:pos + counts[g]]; pos += int(counts[g])
        mx = int(v.max()) if v.size else 0
        npl = (mx.bit_length() + 7) // 8
        comps = [zlib.compress(((v >> np.uint32(8 * p)) & np.uint32(255)).astype(np.uint8).tobytes(), 9) for p in range(npl)]
        table += struct.pack('<IB', int(counts[g]), npl) + b''.join(struct.pack('<I', len(c)) for c in comps)
        body += b''.join(comps)
    return table, body


def encode(Z, ci0, rj0):
    nrows, ncols = Z.shape
    u = Z.view(np.uint32)
    assert np.isfinite(Z).all(), 'NaN/inf in DEM'
    assert not (u == np.uint32(0x80000000)).any(), 'negative zero in DEM'
    q = np.rint(Z.astype(np.float64) * LAT_SCALE).astype(np.int32)
    assert int(np.abs(q).max()) < (1 << 24)
    w = q.astype(np.float32) / np.float32(LAT_SCALE)
    assert (w.astype(np.float64) * LAT_SCALE == q).all()
    d64 = ordered(Z).astype(np.int64) - ordered(w).astype(np.int64)
    d = d64.astype(np.int32)
    assert (d.astype(np.int64) == d64).all()
    del d64
    back = unordered((ordered(w).astype(np.int64) + d).astype(np.int32))
    assert (back.view(np.uint32) == u).all(), 'transform not invertible'
    del back, w
    # layer 1
    r = (q - med(q)).astype(np.int32)
    act = np.abs(shift(r, 0, 1)) + np.abs(shift(r, 1, 0))
    ctx = np.digitize(act, CTX_BINS).astype(np.uint8)
    del act
    zr = zigzag(r)
    del r
    t1, b1 = streams(zr, ctx, len(CTX_BINS) + 1)
    del zr, ctx
    # layer 2
    aq = np.abs(q)
    bl = np.zeros(q.shape, np.int32)
    for k in range(NBL - 1):
        bl += (aq >= (1 << k))                       # bit_length(|q|) for |q| < 2^(NBL-1)
    assert int(aq.max()) < (1 << (NBL - 1))
    nz = (d != 0)
    grp = (((bl * 2 + (q < 0)) * 2 + shift(nz, 0, 1)) * 2 + shift(nz, 1, 0)).astype(np.uint8)
    del bl, nz, aq, q
    t2, b2 = streams(zigzag(d), grp, NGROUPS)
    head = MAGIC + struct.pack('<IIiiI', ncols, nrows, ci0, rj0, 60)
    head += struct.pack('<I', len(CTX_BINS)) + struct.pack('<%di' % len(CTX_BINS), *CTX_BINS)
    blob = head + t1 + t2 + b1 + b2
    return blob + hashlib.sha256(blob).digest(), len(b1), len(b2)



def footprint(inv):
    x0, x1, y0, y1 = inv['canvas']
    lo = []; la = []
    for x in (x0, x1):
        for y in (y0, y1):
            lo.append(inv['lon'][0] + inv['lon'][1] * x + inv['lon'][2] * y)
            la.append(inv['lat'][0] + inv['lat'][1] * x + inv['lat'][2] * y)
    ci0 = int(math.floor((min(lo) + 180) * 60 - 0.5)) - MARGIN; ci1 = int(math.ceil((max(lo) + 180) * 60 - 0.5)) + MARGIN
    rj0 = int(math.floor((min(la) + 90) * 60 - 0.5)) - MARGIN; rj1 = int(math.ceil((max(la) + 90) * 60 - 0.5)) + MARGIN
    return ci0, ci1, rj0, rj1


def load_grid(ci0, ci1, rj0, rj1):
    """The float32 NOAA grid for this window, with missing cells represented as zero."""
    from dem import get_dem
    west = ci0 / 60 - 180; east = (ci1 + 1) / 60 - 180; south = rj0 / 60 - 90; north = (rj1 + 1) / 60 - 90
    lon, lat, Z = get_dem(west + 0.001, east - 0.001, south + 0.001, north - 0.001)
    i0 = int(round((lon[0] + 180) * 60 - 0.5)); j0 = int(round((lat[0] + 90) * 60 - 0.5))
    Zc = Z[rj0 - j0:rj1 - j0 + 1, ci0 - i0:ci1 - i0 + 1]
    return np.ascontiguousarray(np.where(np.isnan(Zc), 0, Zc).astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default=os.environ.get('TERRAIN_DEM_CACHE', os.path.join(DEM_DIR, 'cache')))
    ap.add_argument('--out', default=POLICY_DIR)
    ap.add_argument('--calibration', default=os.path.join(POLICY_DIR, 'calibration.json'))
    a = ap.parse_args()
    os.environ['TERRAIN_DEM_CACHE'] = a.cache
    sys.path.insert(0, DEM_DIR)
    os.makedirs(a.out, exist_ok=True)
    with open(a.calibration, encoding='utf8') as source:
        cal = json.load(source)
    stats = {}
    for t in ('3', '1'):
        inv = {'lon': cal['theatres'][t]['inv_lon'], 'lat': cal['theatres'][t]['inv_lat'], 'canvas': CANVAS[t]}
        ci0, ci1, rj0, rj1 = footprint(inv)
        Z = load_grid(ci0, ci1, rj0, rj1)
        blob, n1, n2 = encode(Z, ci0, rj0)
        fn = 'dem_%s.bin' % NAMES[t]
        open(os.path.join(a.out, fn), 'wb').write(blob)
        st = {'file': fn, 'bytes': len(blob), 'layer1_bytes': n1, 'layer2_bytes': n2, 'cols': int(Z.shape[1]), 'rows': int(Z.shape[0]), 'ci0': ci0, 'rj0': rj0,
              'lon_range': [round((ci0 + 0.5) / 60 - 180, 4), round((ci1 + 0.5) / 60 - 180, 4)], 'lat_range': [round((rj0 + 0.5) / 60 - 90, 4), round((rj1 + 0.5) / 60 - 90, 4)],
              'land_cells': int((Z > 0).sum()), 'sea_cells': int((Z < 0).sum()), 'sha256': hashlib.sha256(blob).hexdigest(),
              'grid_float32_sha256': hashlib.sha256(Z.tobytes()).hexdigest()}
        stats[NAMES[t]] = st
        print(json.dumps({NAMES[t]: st}, indent=1), flush=True)
        del Z, blob
    open(os.path.join(a.out, 'calibration.json'), 'w', newline='\n').write(json.dumps(cal, indent=1, sort_keys=True) + '\n')
    open(os.path.join(a.out, 'README.txt'), 'w', newline='\n').write(readme(stats))


def readme(stats):
    total = sum(s['bytes'] for s in stats.values())
    return ('Terrain policy data (public data only; nothing game-derived).\n\n'
            'dem_india.bin, dem_america.bin\n'
            '  LOSSLESS crops of NOAA NCEI ETOPO 2022 60 arc-second Global Relief Model, surface elevation (z), doi:10.25921/fd45-gt74 (public domain),\n'
            '  https://www.ngdc.noaa.gov/thredds/ncss/grid/global/ETOPO2022/60s/60s_surface_elev_netcdf/ETOPO_2022_v1_60s_N90W180_surface.nc\n'
            '  cropped to the fixed India (theatre 3) / America (theatre 1) canvas footprints + 40 cell margin. Every float32 cell is stored bit-exactly\n'
            '  (missing cells become zero): 1/16 m lattice integer + float32-ulp offset, MED-predicted, context-split zlib.\n'
            '  Layout and transform: tools/terrain_paint/prepare_dem.py. Decoded by src/terrain/art/dem.rs (include_bytes!, SHA-256 trailer verified at load).\n'
            '  Total %d bytes for both grids. The encoder checks exact float32 reconstruction; grid_float32_sha256 below identifies\n'
            '  each complete NetCDF-derived input grid independently of its compressed representation.\n\n' % total +
            'calibration.json\n'
            '  Authored world <-> lon/lat affine fits for theatres 1 and 3 and a 41-point metres -> height-byte curve.\n'
            '  These numeric calibration constants contain no game texture or heightmap pixels.\n\n'
            'Regenerate (dev only): python -B tools/terrain_paint/prepare_dem.py [--cache DIR]   (NetCDF tiles are fetched from NOAA when absent)\n\n'
            'Measured (prepare_dem.py output; grid_float32_sha256 = SHA-256 of the little-endian float32 row-major NetCDF-derived grid):\n' + json.dumps(stats, indent=1) + '\n')


if __name__ == '__main__':
    main()
