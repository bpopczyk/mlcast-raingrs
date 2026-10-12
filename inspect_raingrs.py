import tarfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import xarray as xr

from pyproj import CRS, Transformer
import zarr

# ------------------------------------------------------------------
# INPUT
# ------------------------------------------------------------------

tar_path = (
    r"C:\Users\bpopczyk\programy_C\MLCast\data\2020\01"
    r"\2020-01-01_grs_10.tar.gz"
)

zarr_path = (
    r"C:\Users\bpopczyk\programy_C\MLCast\data"
    r"\RainGRS_1day.zarr"
)

# ------------------------------------------------------------------
# READ RAINGRS
# ------------------------------------------------------------------

data_list = []
time_list = []

with tarfile.open(tar_path, "r:gz") as tar:

    members = sorted(tar.getnames())

    for name in members:

        timestamp = pd.to_datetime(
            name[:12],
            format="%Y%m%d%H%M"
        )

        import io

        with tar.extractfile(name) as f:
        
            raw = f.read()
        
            with h5py.File(
                io.BytesIO(raw),
                "r"
            ) as h:

                rain = h["dataset00"]["data0"]["data"][:]

                # nodata -> NaN
                rain = rain.astype(np.float32)
                rain[rain == -999] = np.nan

                # --------------------------------------------------
                # geometry (only once)
                # --------------------------------------------------

                if len(data_list) == 0:

                    xsize = int(h["where"].attrs["xsize"][0])
                    ysize = int(h["where"].attrs["ysize"][0])

                    xll = float(h["where"].attrs["xllcorner"][0])
                    yll = float(h["where"].attrs["yllcorner"][0])

                    dx = float(h["where"].attrs["xscale"][0])
                    dy = float(h["where"].attrs["yscale"][0])

                    projdef = (
                        h["where"].attrs["projdef"]
                        .decode("utf-8")
                    )

                # --------------------------------------------------
                # flip Y to match common GeoRaster convention
                # --------------------------------------------------

                rain = np.flipud(rain)

                data_list.append(rain)
                time_list.append(timestamp)

# ------------------------------------------------------------------
# STACK
# ------------------------------------------------------------------

data = np.stack(data_list)

print("data shape:", data.shape)

# ------------------------------------------------------------------
# COORDINATES
# ------------------------------------------------------------------

x = xll + dx * np.arange(xsize)

y = yll + dy * np.arange(ysize)
y = y[::-1]

# ------------------------------------------------------------------
# LAT/LON
# ------------------------------------------------------------------

xx, yy = np.meshgrid(x, y)

crs_proj = CRS.from_proj4(projdef)

transformer = Transformer.from_crs(
    crs_proj,
    "EPSG:4326",
    always_xy=True,
)

lon, lat = transformer.transform(xx, yy)

# ------------------------------------------------------------------
# DATASET
# ------------------------------------------------------------------

ds = xr.Dataset(
    data_vars={
        "depth": (
            ("time", "y", "x"),
            data,
        )
    },
    coords={
        "time": time_list,
        "x": x,
        "y": y,
        "lat": (
            ("y", "x"),
            lat.astype(np.float64),
        ),
        "lon": (
            ("y", "x"),
            lon.astype(np.float64),
        ),
    },
)

# ------------------------------------------------------------------
# COORD ATTRS
# ------------------------------------------------------------------

ds["x"].attrs = {
    "units": "m",
    "standard_name": "projection_x_coordinate",
    "axis": "X",
}

ds["y"].attrs = {
    "units": "m",
    "standard_name": "projection_y_coordinate",
    "axis": "Y",
}

ds["lat"].attrs = {
    "long_name": "Latitude",
    "standard_name": "latitude",
    "units": "degrees_north",
}

ds["lon"].attrs = {
    "long_name": "Longitude",
    "standard_name": "longitude",
    "units": "degrees_east",
}

# ------------------------------------------------------------------
# VARIABLE ATTRS
# ------------------------------------------------------------------

ds["depth"].attrs = {
    "units": "mm",
    "long_name": "Accumulated precipitation depth",
    "standard_name": "precipitation_amount",
    "grid_mapping": "crs",
}
# ------------------------------------------------------------------
# CRS
# ------------------------------------------------------------------

ds["crs"] = xr.DataArray(
    np.float32(0),
    attrs={
        "grid_mapping_name": "transverse_mercator",
        "latitude_of_projection_origin": 0.0,
        "longitude_of_central_meridian": 19.0,
        "scale_factor_at_central_meridian": 0.9993,
        "false_easting": 500000.0,
        "false_northing": -5300000.0,
        
        "semi_major_axis": 6378137.0,
        "inverse_flattening": 298.257223563,
        "proj4": projdef,
        "crs_wkt": crs_proj.to_wkt(),
        "spatial_ref": crs_proj.to_wkt(),
        "GeoTransform": (
            f"{x[0]} {dx} 0.0 "
            f"{y[0]} 0.0 {-dy}"
        ),
        "where_xsize": xsize,
        "where_ysize": ysize,
        "where_xscale": dx,
        "where_yscale": dy,
    },
)

# link variables to CRS

ds["depth"].attrs["grid_mapping"] = "crs"
ds["lat"].attrs["grid_mapping"] = "crs"
ds["lon"].attrs["grid_mapping"] = "crs"

# ------------------------------------------------------------------
# GLOBAL ATTRS
# ------------------------------------------------------------------

ds.attrs.update(
    {
        "license": "CC-BY-4.0",

        "mlcast_created_on": pd.Timestamp.now().isoformat(),

        "mlcast_created_by":
            "Barbara Popczyk <barbara.popczyk@imgw.pl>",

        "mlcast_created_with":
            "https://github.com/barbara-popczyk/raingrs-to-geozarr@v0.1.0",

        "mlcast_dataset_version": "0.1.0",

        "mlcast_dataset_validator_version": "0.3.0",

        "mlcast_dataset_identifier":
            "pl-imgw-radar_precipitation",
    }
)
# ------------------------------------------------------------------
# CHUNKS
# ------------------------------------------------------------------

ds = ds.chunk(
    {
        "time": 1,
        "y": ysize,
        "x": xsize,
    }
)

# ------------------------------------------------------------------
# SAVE
# ------------------------------------------------------------------

import shutil
from pathlib import Path

p = Path(zarr_path)

if p.exists():
    shutil.rmtree(p)

ds.to_zarr(
    zarr_path,
    mode="w",
    consolidated=True,
    zarr_format=2,
)

from pathlib import Path

p = Path(
    r"C:\Users\bpopczyk\programy_C\MLCast\data\RainGRS_1day.zarr"
)

print("zarr.json :", (p / "zarr.json").exists())
print(".zgroup   :", (p / ".zgroup").exists())
print(".zattrs   :", (p / ".zattrs").exists())

print()
print(ds["crs"])
print()
print(ds["crs"].attrs)
