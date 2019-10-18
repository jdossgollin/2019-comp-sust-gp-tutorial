"""
See if we can read the data in
"""
import os
import copy
from functools import reduce
from typing import List

import geopandas as gpd
import GPy
from IPython.display import display
import matplotlib.pyplot as plt
import navpy
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "dat"))
ANALYSIS_DATE = pd.Timestamp("2014-07-25")


def get_data(fnames: List[str]) -> pd.DataFrame:
    """Read in the raw data from hd5
    """

    def read_file(fname: str) -> pd.DataFrame:
        """Read in a single file
        """
        raw_data = (
            pd.read_hdf(fname).to_frame().reset_index(level=[0, 1]).loc[ANALYSIS_DATE]
        )
        raw_data["date"] = raw_data.index
        return raw_data

    raw_dfs = [read_file(fname) for fname in fnames]
    clean_data = reduce(
        lambda left, right: pd.merge(left, right, how="inner", on=["lat", "lon"]),
        raw_dfs,
    )
    try:
        clean_data.drop("date_x", axis=1, inplace=True)
        clean_data.drop("date_y", axis=1, inplace=True)
    except KeyError:
        print("Columns not found.")
    return clean_data


fnames = [
    os.path.join(DATA_DIR, file)
    for file in [
        "gt-contest_precip-14d-1948-2018.h5",
        "gt-contest_tmax-14d-1979-2018.h5",
        "gt-contest_tmin-14d-1979-2018.h5",
    ]
]
df_full = get_data(fnames)

# some geopandas magic
geometry = [Point((x - 360.0, y)) for (x, y) in zip(df_full["lon"], df_full["lat"])]
gdf = gpd.GeoDataFrame(copy.copy(df_full), geometry=geometry)
world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
world = world.to_crs({"init": "epsg:4326"})  # world.to_crs(epsg=3395) would also work


def plot_var(var: str, geo_df: gpd.GeoDataFrame) -> None:
    """Plot the variable
    """
    minx, miny, maxx, maxy = gdf.total_bounds
    ax = geo_df.plot(
        ax=world.plot(figsize=(10, 6), alpha=0.3, color="grey"),
        marker="o",
        c=geo_df[var],
        markersize=100,
    )
    ax.set_xlim(minx - 2.0, maxx + 2.0)
    ax.set_ylim(miny - 2.0, maxy + 2.0)
    ax.set_title(f"{var} on {ANALYSIS_DATE}")
    return ax


# The projection from latitutde/longtiude to meters must be done at a
# specific reference point. We choose a reference location somewhere
# in the middle-ish of the grid
lat_ref = (gdf.total_bounds[3] + gdf.total_bounds[1]) / 2.0
lon_ref = (gdf.total_bounds[2] + gdf.total_bounds[0]) / 2.0

# Perform the conversion for data in the full dataset
coord_m = np.array(
    [
        navpy.lla2ned(
            lat=lat,
            lon=lon - 360.0,
            alt=0.0,
            lat_ref=lat_ref,
            lon_ref=lon_ref,
            alt_ref=0.0,
        )
        / 1000.0
        for (lat, lon) in zip(df_full["lat"], df_full["lon"])
    ]
)

# Set the kilometers north and south
df_full["km_n"] = pd.Series(coord_m[:, 0])
df_full["km_e"] = pd.Series(coord_m[:, 1])

plt.scatter(x=df_full["km_e"], y=df_full["km_n"])

# First, we examine the mean of the distribution. We could incorporate
# this into a prior mean funciton, or we can center the data and use
# a zero mean function.
print(df_full[["tmin", "tmax", "precip"]].mean())

# center the data
df_centered = df_full.copy()
df_centered['precip'] = df_centered['precip'] - df_centered['precip'].mean()
df_centered['tmax'] = df_centered['tmax'] - df_centered['tmax'].mean()
df_centered['tmin'] = df_centered['tmin'] - df_centered['tmin'].mean()

analysis_datum = 'tmin'


# Split the data with 33% training and 67% testing
df_shuffle = df_centered.sample(frac=1.0, replace=False, axis=0)
df_split = np.array_split(df_shuffle, [df_shuffle.shape[0] // 3])
df_split[0].shape, df_split[1].shape

df_train = df_split[0]
df_test = df_split[1]

# Plot the training and testing locations
plt.scatter(x=df_train['km_e'], y=df_train['km_n'], c=df_train[analysis_datum])
plt.title("Training Locations")
plt.show()

# Plot various kernels
ax = plt.axes()
vis_kernel1 = GPy.kern.RBF(input_dim=1, variance=3., lengthscale=1)
vis_kernel2 = GPy.kern.RBF(input_dim=1, variance=3., lengthscale=2)
vis_kernel3 = GPy.kern.RBF(input_dim=1, variance=0.5, lengthscale=1)
vis_kernel1.plot(ax=ax, color = 'r', label='$\sigma^2=3$, $\ell=1$')
vis_kernel2.plot(ax=ax, color = 'b', label='$\sigma^2=3$, $\ell=2$')
vis_kernel3.plot(ax=ax, color = 'g', label='$\sigma^2=0.5$, $\ell=1$')
plt.legend()

# Initilaize a Radial Basis funciton kernel with two dimensions and set
# the kernel parameters to something "reasonable"
kernel = GPy.kern.RBF(input_dim=2, variance=10., lengthscale=[100., 150.], ARD=True)
display(kernel)
kernel.plot()

# Initialize the model we'll use with random values
X = np.array([df_train['km_e'], df_train['km_n']]).T
y = np.atleast_2d(np.array(df_train[analysis_datum])).T
m = GPy.models.GPRegression(X, y, kernel)
Xpred = np.array([df_test['km_e'], df_test['km_n']]).T

# Predict the values at the new locations in the "testing" dataset
ypred, yvar = m.predict(Xnew=Xpred)

# The ground truth precipitation/temperature data
ytest = np.atleast_2d(np.array(df_test[analysis_datum])).T
loss_test = ((ytest - ypred)**2).sum()
print("Predictive loss:", loss_test)

# now optimize the model
m.optimize_restarts(messages=True, num_restarts=10)

# Now we can make predictions using the optimized kernel parameters. 
Xpred = np.array([df_test['km_e'], df_test['km_n']]).T

# Predict the values at the new locations in the "testing" dataset
ypred, yvar = m.predict(Xnew=Xpred)

# The ground truth precipitation/temperature data
ytest = np.atleast_2d(np.array(df_test[analysis_datum])).T

loss_test = ((ytest - ypred)**2).sum()
print("Predictive loss:", loss_test, "(hopefully smaller)") #

# Plot residuals
fig, axes = plt.subplots(nrows=1, ncols=2)
ax = axes[0]
ax.scatter(x=Xpred[:, 0], y=Xpred[:, 1], c=ypred.reshape(-1,), cmap="viridis")
ax.set_title("Predicted")
ax = axes[1]
resid = ypred - ytest
ax.scatter(x=Xpred[:, 0], y=Xpred[:, 1], c=resid.reshape(-1,), cmap="PuOr")
ax.set_title("Residual")
plt.show()