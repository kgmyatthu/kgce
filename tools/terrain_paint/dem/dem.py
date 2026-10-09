"""ETOPO 2022 60-arcsecond surface elevation, fetched in 10x10 degree NCSS tiles and cached as .nc files."""
import math, os, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.io import netcdf_file
CACHE = os.environ.get('TERRAIN_DEM_CACHE', os.path.join(os.path.dirname(__file__), 'cache'))

URL = ('https://www.ngdc.noaa.gov/thredds/ncss/grid/global/ETOPO2022/60s/60s_surface_elev_netcdf/'
       'ETOPO_2022_v1_60s_N90W180_surface.nc?var=z&north={n}&south={s}&west={w}&east={e}&accept=netcdf3')
RES = 1.0 / 60.0


def _tile_path(lon0, lat0):
    return os.path.join(CACHE, f'etopo60_lon{lon0:+04d}_lat{lat0:+03d}.nc')


def _fetch(lon0, lat0):
    p = _tile_path(lon0, lat0)
    if os.path.exists(p) and os.path.getsize(p) > 1000:
        return p
    os.makedirs(CACHE, exist_ok=True)
    inset = 0.004  # NCSS asserts on bounds that coincide with grid-cell edges; stay inside the first/last cell centre
    url = URL.format(n=lat0 + 10 - inset, s=lat0 + inset, w=lon0 + inset, e=lon0 + 10 - inset)
    last = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(p + '.part', 'wb') as f:
                f.write(r.read())
            os.replace(p + '.part', p)
            return p
        except Exception as ex:  # network hiccup: retry
            last = ex; time.sleep(2 * (attempt + 1))
    raise RuntimeError(f'fetch failed {url}: {last}')


def get_dem(west, east, south, north):
    """Return (lon[ncol], lat[nrow ascending], z[nrow,ncol] float32) covering the box (rounded out to 10-deg tiles, then cropped)."""
    lon_tiles = list(range(int(math.floor(west / 10)) * 10, int(math.ceil(east / 10)) * 10, 10))
    lat_tiles = list(range(max(-90, int(math.floor(south / 10)) * 10), min(90, int(math.ceil(north / 10)) * 10), 10))
    jobs = [(a, b) for a in lon_tiles for b in lat_tiles]
    with ThreadPoolExecutor(6) as ex:
        paths = list(ex.map(lambda ab: _fetch(*ab), jobs))
    i0 = int(round((lon_tiles[0] + 180) * 60)); j0 = int(round((lat_tiles[0] + 90) * 60))
    ncol = len(lon_tiles) * 600; nrow = len(lat_tiles) * 600
    Z = np.full((nrow, ncol), np.nan, np.float32)
    for (a, b), p in zip(jobs, paths):
        f = netcdf_file(p, mmap=False)
        lon = np.array(f.variables['lon'][:], float); lat = np.array(f.variables['lat'][:], float)
        z = np.array(f.variables['z'][:], np.float32); f.close()
        z[z < -9999] = np.nan
        ci = np.round((lon + 180) / RES - 0.5).astype(int) - i0
        ri = np.round((lat + 90) / RES - 0.5).astype(int) - j0
        if lat[0] > lat[-1]:  # descending latitude in file
            pass
        ok_c = (ci >= 0) & (ci < ncol); ok_r = (ri >= 0) & (ri < nrow)
        Z[np.ix_(ri[ok_r], ci[ok_c])] = z[np.ix_(np.where(ok_r)[0], np.where(ok_c)[0])]
    lon = (np.arange(ncol) + i0 + 0.5) * RES - 180
    lat = (np.arange(nrow) + j0 + 0.5) * RES - 90
    return lon, lat, Z

