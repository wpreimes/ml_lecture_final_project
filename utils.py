import numpy as np
np.random.seed(123)
import pandas as pd
import os
from pathlib import Path
import re
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from glob import glob


original_vars = ['lai_hv', 'lai_lv', 'tp', 'd2m', 'skt', 'stl1', 'stl2', 'stl3', 'stl4', 't2m', 'evabs', 'evaow', 'evatc', 'evavt', 'pev', 'ro', 'es', 'ssro', 'sro']

pretty_vars = [
    "Leaf Area Index (high vegetation)",
    "Leaf Area Index (low vegetation)",
    "Total precipitation",
    "2-metre dewpoint temperature",
    "Skin temperature",
    "Soil temperature level 1 (0–7 cm)",
    "Soil temperature level 2 (7–28 cm)",
    "Soil temperature level 3 (28–100 cm)",
    "Soil temperature level 4 (100–289 cm)",
    "2-metre temperature",
    "Evaporation from bare soil",
    "Evaporation from open water surfaces (excluding oceans)",
    "Evaporation from top of canopy",
    "Evaporation from vegetation transpiration",
    "Potential evaporation",
    "Total runoff",
    "Snow evaporation",
    "Sub-surface runoff",
    "Surface runoff"
]

_l = Line2D([], [], linestyle='none')


def _marker_on_map(lat, lon, ax, name=None, other_points=None):
    # Europe view
    ax.set_extent([-25, 45, 34, 72], crs=ccrs.PlateCarree())

    # Simple, clean map features
    ax.add_feature(cfeature.OCEAN, facecolor="#dbe9f4")
    ax.add_feature(cfeature.LAND, facecolor="#f2efe9")
    ax.add_feature(cfeature.COASTLINE, linewidth=1)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.8)

    other_points = other_points or []
    for (_lon, _lat) in other_points:
            ax.plot(
            _lon, _lat,
            marker=".",
            color="black",
            markersize=8,
            markeredgecolor="black",
            transform=ccrs.PlateCarree(),
        )
        
    # Big marker
    ax.plot(
        lon, lat,
        marker="o",
        color="red",
        markersize=14,
        markeredgecolor="black",
        transform=ccrs.PlateCarree(),
    )

        # Optional label
    ax.text(
        lon + 1, lat + 3,
        f"y={lat:.2f}, x={lon:.2f}",
        transform=ccrs.PlateCarree(),
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.7, boxstyle="round")
    )
    
        
    ax.set_title(name, fontsize=14)
    return ax


def read_reanalyis_file(path):
    # read data from the passed file
    reanalysis = pd.read_csv(path)
    reanalysis['date_time'] = pd.to_datetime(reanalysis['Unnamed: 0'])
    reanalysis = reanalysis.set_index('date_time')
    reanalysis = reanalysis.drop(columns = ['Unnamed: 0'])
    for col in ['swvl2', 'swvl3', 'swvl4']:
        if col in reanalysis.columns:
            reanalysis = reanalysis.drop(columns=col)
    reanalysis = reanalysis.rename(columns={'swvl1': "reanalysis SM"})
    # Rainfall is ACCUMLATED over 24 hours, we use the value at 0:00 from the prev. day
    precip = (
        reanalysis[reanalysis.index.hour == 0]
        .assign(index=lambda x: x.index - pd.Timedelta(days=1))
        .set_index("index")[["tp"]]
        .asfreq("D")
    )
    reanalysis = reanalysis.resample('D').mean()
    reanalysis['tp'] = precip
    reanalysis = reanalysis.rename(columns=dict(zip(original_vars, pretty_vars)))
    return reanalysis

    
class ReanalysisReader:
    def __init__(self, path="extra_era5land_europe/era5land_*.csv"):
        """
        path: str or Path
            Path where the era5 time series files are stored
        shuffle: bool, optional (default: False)
            Shuffle the file list, so that read_fid does not use the
            sorted list.
        """
        
        self.path = Path(path)
        self.files = np.array(sorted(glob(str(self.path))))
        
        lons = []
        lats = []
        for f in self.files:
            lon, lat = self.parse_lon_lat(os.path.basename(f))
            lons.append(lon)
            lats.append(lat)

    def parse_lon_lat(self, filename: str) -> tuple[float, float]:
        pattern = r"lon=(-?\d+\.?\d*)_lat=(-?\d+\.?\d*)"
        match = re.search(pattern, filename)
        lon, lat = match.group(1), match.group(2)
        return float(lon), float(lat)
    
    def __len__(self):
        return len(self.files)

    def read_fid(self, fid):
        # Read data from the ith file
        fname = self.files[fid]
        return read_reanalyis_file(fname)
    
    

def _load_data(location):
    start = '2015-01-01'
    end = '2020-12-31'

    pattern = r"(.*?)_lat=([-+]?\d*\.?\d+)_lon=([-+]?\d*\.?\d+)"
    other_coords = []

    for fname in os.listdir("data"):
        match = re.search(pattern, fname)
        name = match.group(1)
        _lat  = float(match.group(2))
        _lon  = float(match.group(3))
        if name == location:
            lat, lon = _lat, _lon
            path = Path(f"data/{fname}")
        else:
            other_coords.append((_lon, _lat))
            
    path = Path(glob(f"data/{location}_*")[0])

    insitu = pd.read_csv(path / 'insitu.csv')
    insitu['date_time'] = pd.to_datetime(insitu['date_time'])
    insitu = insitu.set_index('date_time')
    insitu = insitu.loc[insitu["soil_moisture_flag"] == 'G', ['soil_moisture']]
    insitu = insitu.rename(columns={'soil_moisture': 'in situ SM'}).resample('D').mean()
    insitu = insitu.loc[start:end, :]

    reanalysis = read_reanalyis_file(path / 'reanalysis.csv')
    reanalysis = reanalysis.loc[start:end, :]

    satellite = pd.read_csv(path / 'satellite.csv')
    satellite['date_time'] = pd.to_datetime(satellite['Unnamed: 0'])
    satellite = satellite.set_index('date_time')
    satellite = satellite.drop(columns = ['Unnamed: 0'])
    satellite = satellite[['sm']].rename(columns={'sm': 'satellite SM'})
    satellite = satellite.loc[start:end, :]
    
    return insitu, reanalysis, satellite, float(lat), float(lon), other_coords


def min_max_scale(s: pd.Series, out_min: float, out_max: float) -> pd.Series:
    s_min, s_max = s.min(), s.max()
    return out_min + (s - s_min) * (out_max - out_min) / (s_max - s_min)
