import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from heat_loss_utils import compute_heat_losses
from scipy.interpolate import griddata
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- Page config ---
st.set_page_config(page_title="♨️ Heat Loss Components", layout="wide")
st.title("🔥 Pool Heat Loss Breakdown by Component")

# Sidebar inputs
st.sidebar.header("Input Parameters")

# --- Input sliders ---
helideck_diameter = st.sidebar.slider(
    "Helideck Diameter (m)", 
    min_value=10.0, 
    max_value=25.0, 
    value=12.5, 
    step=0.5
)
collector_ratio = st.sidebar.slider("Collector Area Ratio", 0.1, 1.0, 0.6)
collector_area = np.pi * (helideck_diameter / 2) ** 2 * collector_ratio

st.sidebar.markdown(f"**Collector Area:** {collector_area:.1f} m²")

# --- Mini helideck and collector visualization ---
def plot_helideck_sidebar(diameter, collector_area):
    fig, ax = plt.subplots(figsize=(2.5, 2.5), dpi=100)
    fig.patch.set_alpha(0)  # Transparent background

    helideck_radius = diameter / 2
    collector_radius = np.sqrt(collector_area / np.pi)

    # Draw collector (yellow filled circle)
    collector = plt.Circle((0, 0), collector_radius, color='yellow', alpha=0.8, label="Solar Collector")
    ax.add_patch(collector)

    # Draw helideck (black outline)
    helideck = plt.Circle((0, 0), helideck_radius, color='black', fill=False, linewidth=2, label="Helideck")
    ax.add_patch(helideck)

    # Draw transparent "H" in center
    ax.text(0, 0, "H", color="black", fontsize=80, ha='center', va='center', alpha=0.2, weight='bold')

    # Labels
    ax.text(0, collector_radius * 0.0, "Solar\nCollector", color="black", fontsize=8, ha='center', va='center', weight='bold')
    ax.text(0, helideck_radius * 1.05, "Helideck", color="black", fontsize=8, ha='center', va='bottom', weight='bold')

    ax.set_xlim(-helideck_radius * 1.2, helideck_radius * 1.2)
    ax.set_ylim(-helideck_radius * 1.2, helideck_radius * 1.2)
    ax.set_aspect('equal')
    ax.axis('off')

    return fig

# Show in sidebar
st.sidebar.pyplot(plot_helideck_sidebar(helideck_diameter, collector_area))




helideck_area = collector_area
collector_efficiency = 0.7
pool_temp = st.sidebar.slider("Desired Pool Temp (°C)", 20, 35, 28)
pool_area = st.sidebar.slider("Pool Area (m²)", 10, 100, 50)
pool_depth = 1.5
night_hours = 12
cover_used = st.sidebar.checkbox("Use Pool Cover at Night", value=True)

shielding = st.sidebar.selectbox(
    "Wind speed at pool surface relative to 10 m reference wind speed", [
        "Open exposure (70%) – e.g. Open deck without any obstructions",
        "Partly shielded (40%) – some walls or windbreaks",
        "Recessed or surrounded (15%) – Recessed or large wind breaks",
        "Highly shielded (5%) – Partly enclosed"
    ],
    index=1  # Default is the second option
)
shielding_factor = {
    "Open exposure (70%) – e.g. Open deck without any obstructions": 0.7,
    "Partly shielded (40%) – some walls or windbreaks": 0.4,
    "Recessed or surrounded (15%) – Recessed or large wind breaks": 0.15,
    "Highly shielded (5%) – Partly enclosed": 0.05
}[shielding]

month = st.sidebar.selectbox("Select Month", [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
])
# --- Load climate data ---
@st.cache_data
def load_data():
    return pd.read_csv("climate_data_sea.csv")

df = load_data()
lat = df["lat"].values
lon = df["lon"].values

ghi_cols = df.columns[df.columns.str.startswith("ghi_")]
mask = (df['lat'] < -65) | ((df['lat'] > 60) & (df['lon'].between(-60, -20)))
df.loc[mask, ghi_cols] *= 0.5

# Clean polar outliers
ghi_cols = df.columns[df.columns.str.startswith("ghi_")]
mask = (df['lat'] < -65) | ((df['lat'] > 60) & (df['lon'].between(-60, -20)))
df.loc[mask, ghi_cols] *= 0.5

# Climate values
tmin = df[f"tmin_{month}"].values
tmax = df[f"tmax_{month}"].values
tavg = df.get(f"tavg_{month}", (tmin + tmax) / 2).values
T_day = (tavg + tmax) / 2
T_night = (tavg + tmin) / 2
wind = df[f"ws10m_{month}"].values
wind_day = wind * shielding_factor
wind_night = 0.8 * wind * shielding_factor
rh = df[f"rh_{month}"].values
rh_day = rh
rh_night = 1.1 * rh

# --- Compute losses ---
loss = compute_heat_losses(
    pool_temp, pool_area, pool_depth, T_day, T_night,
    wind_day, wind_night, rh_day, rh_night, night_hours, cover_used
)

rad_loss = loss["rad_day"] + loss["rad_night"]
evap_loss = loss["evap_day"] + loss["evap_night"]
conv_loss = loss["conv_day"] + loss["conv_night"]
total_loss = loss["Q_day"] + loss["Q_night"]


# --- Plot function ---
def plot_loss_map(data, title, cmap, lon, lat):
    # Create interpolation grid
    lon_grid = np.linspace(min(lon), max(lon), 200)
    lat_grid = np.linspace(min(lat), max(lat), 150)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

    # Interpolate to grid
    grid = griddata((lon, lat), data, (lon_mesh, lat_mesh), method='linear')

    # Compute vmin and vmax for colorbar, excluding Antarctica and Greenland
    mask = (lat > -65) & ~((lat > 60) & (lon > -60) & (lon < -20))
    capped_max = np.nanpercentile(data[mask], 99)  # e.g. 99th percentile
    capped_min = 0

    # --- PLOT ---
    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Use clipped grid for color values but plot full grid
    cf = ax.contourf(
        lon_mesh, lat_mesh,
        np.clip(grid, capped_min, capped_max),  # clip for color scale only
        levels=100, cmap=cmap, vmin=capped_min, vmax=capped_max
    )

    # Add contours (can use unclipped grid here)
    cs = ax.contour(lon_mesh, lat_mesh, grid, colors='black', linewidths=0.3)
    ax.clabel(cs, fmt='%d', fontsize=8)

    # Add map features
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.set_title(title)

    # Colorbar
    cbar = fig.colorbar(cf, ax=ax, orientation='vertical', shrink=0.8, pad=0.05)
    cbar.set_label("kWh/day")

    return fig
# --- Show plots ---
col1, col2 = st.columns(2)
with col1:
    st.pyplot(plot_loss_map(rad_loss, "Radiation Loss per Day", "jet",lon,lat))
    st.pyplot(plot_loss_map(evap_loss, "Evaporation Loss per Day", "jet",lon,lat))
with col2:
    st.pyplot(plot_loss_map(conv_loss, "Convection Loss per Day", "jet",lon,lat))
    st.pyplot(plot_loss_map(total_loss, "Total Heat Loss per Day", "jet",lon,lat))
