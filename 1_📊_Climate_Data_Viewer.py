import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import cartopy.crs as ccrs
import cartopy.feature as cfeature

st.set_page_config(page_title="Climate Data Viewer", layout="wide")
st.title("📊 Climate Data Viewer")

@st.cache_data
def load_data():
    return pd.read_csv("climate_data_sea.csv")

df = load_data()

# Parameter metadata
parameter_info = {
    "ghi":    {"label": "Global Horizontal Irradiance", "unit": "kWh/m²/day"},
    "tmin":   {"label": "Min Temperature (2m)",         "unit": "°C"},
    "tmax":   {"label": "Max Temperature (2m)",         "unit": "°C"},
    "tavg":   {"label": "Avg Temperature (2m)",         "unit": "°C"},
    "rh":     {"label": "Relative Humidity (2m)",       "unit": "%"},
    "ws10m":  {"label": "Wind Speed (10m)",             "unit": "m/s"},
    "tdew":   {"label": "Dew Point Temperature",        "unit": "°C"},
    "ps":     {"label": "Surface Pressure",             "unit": "kPa"},
}
available_metrics = list(parameter_info.keys())

# Month logic
months_ordered = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
full_to_short = {m: m[:3].lower() for m in months_ordered}
short_to_full = {v: k for k, v in full_to_short.items()}

available_months = [m for m in months_ordered if f"ghi_{m}" in df.columns]
available_months_short = [full_to_short[m] for m in available_months]

# --- Sidebar selectors ---
st.sidebar.header("Select Parameter and Month")
month_short = st.sidebar.radio("Month", available_months_short, horizontal=True)
month = short_to_full[month_short]

metric = st.sidebar.radio(
    "Metric",
    available_metrics,
    format_func=lambda m: parameter_info[m]["label"]
)

# NEW VIEW OPTION: Add a radio button to choose view mode
view_mode = st.sidebar.radio("View Mode", ["Climate Map", "Show Grid Locations (2.5°)"])

# Show the 2.5° x 2.5° grid map
if view_mode == "Show Grid Locations (2.5°)":
    fig, ax = plt.subplots(figsize=(14, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_global()
    ax.coastlines()
    ax.gridlines(draw_labels=True, linewidth=0.2)

    # Create grid
    lats = np.arange(-90, 90 + 2.5, 2.5)
    lons = np.arange(-180, 180 + 2.5, 2.5)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    ax.plot(lon_grid, lat_grid, 'r.', markersize=2.5, transform=ccrs.PlateCarree())

    ax.set_title("World Grid: 2.5° x 2.5° Points")
    st.pyplot(fig)

else:
    # Display the climate map
    column = f"{metric}_{month}"
    unit = parameter_info[metric]["unit"]
    label = parameter_info[metric]["label"]

    st.subheader(f"{label} — {month} [{unit}]")

    if st.checkbox("Show Raw Data Table"):
        st.dataframe(df[["lat", "lon", column]])

    lat = df["lat"].values
    lon = df["lon"].values

    # Apply GHI masking
    ghi_cols = df.columns[df.columns.str.startswith("ghi_")]
    mask = (df['lat'] < -65) | ((df['lat'] > 60) & (df['lon'].between(-60, -20)))
    df.loc[mask, ghi_cols] *= 0.5

    data = df[column].values

    # Interpolation
    lon_grid = np.linspace(min(lon), max(lon), 200)
    lat_grid = np.linspace(min(lat), max(lat), 150)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    grid = griddata((lon, lat), data, (lon_mesh, lat_mesh), method='linear')

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    cf = ax.contourf(lon_mesh, lat_mesh, grid, levels=100, cmap="viridis")

    cs = ax.contour(lon_mesh, lat_mesh, grid, levels=10, colors='black', linewidths=0.5)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.1f")

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.set_title(f"{label} — {month}", fontsize=14)

    cbar = fig.colorbar(cf, ax=ax, shrink=0.7)
    cbar.set_label(unit)

    st.pyplot(fig)
