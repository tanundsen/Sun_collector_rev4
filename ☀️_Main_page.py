
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from heat_loss_utils import compute_heat_losses
from matplotlib.ticker import FormatStrFormatter
from cartopy.feature import NaturalEarthFeature

st.set_page_config(layout="wide")

# Title and logo
col_logo, col_title = st.columns([2, 5])
with col_logo:
    st.image("logo.png", width=360)
with col_title:
    st.title("☀️ Helideck Solar Collector Analysis")

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

# Nighttime parameters
night_hours = 12  # Fixed night time
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


month = st.sidebar.selectbox(
    "Select Month", [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
)

show_large = st.sidebar.checkbox("Show large savings map only")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("climate_data_sea.csv")

df = load_data()

lat = df["lat"].values
lon = df["lon"].values

ghi_cols = df.columns[df.columns.str.startswith("ghi_")]
mask = (df['lat'] < -65) | ((df['lat'] > 60) & (df['lon'].between(-60, -20)))
df.loc[mask, ghi_cols] *= 0.5

# Interpolation grid
lon_grid = np.linspace(min(lon), max(lon), 200)
lat_grid = np.linspace(min(lat), max(lat), 150)
lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

# Climate and energy parameters
value_column = f"ghi_{month}"
tmin = df[f"tmin_{month}"].values
tmax = df[f"tmax_{month}"].values
tavg = df.get(f"tavg_{month}", (tmin + tmax) / 2).values

T_day = (tavg + tmax) / 2
T_night = (tavg + tmin) / 2

ghi = df[value_column].values
wind = df[f"ws10m_{month}"].values
wind_day = wind * shielding_factor
wind_night = 0.8 * wind * shielding_factor
rh = df[f"rh_{month}"].values
rh_day = rh
rh_night = 1.1 * rh

# Heat loss calculation
loss = compute_heat_losses(pool_temp, pool_area, pool_depth, T_day, T_night, wind_day, wind_night, rh_day, rh_night, night_hours, cover_used)
Q_day = loss["Q_day"]
Q_night = loss["Q_night"]
total_loss = Q_day + Q_night

helideck_gain = ghi * helideck_area * collector_efficiency
pool_solar_gain = ghi * pool_area * 0.7
net_pool_heating = np.clip(total_loss - pool_solar_gain, 0, None)
net_saving = np.minimum(helideck_gain, net_pool_heating)
net_saving = np.maximum(net_saving, 0.001)

def plot_map(data, title, cmap, vmin=None, vmax=None, large=False):
    figsize = (12, 7) if large else (8, 5)
    grid = griddata((lon, lat), data, (lon_mesh, lat_mesh), method='linear')

    # Mask out Antarctica and Greenland for colorbar range
    mask = ~(((lat < -65) | ((lat > 60) & (lon >= -60) & (lon <= -20))))
    vmin = vmin if vmin is not None else np.nanmin(data[mask])
    vmax = vmax if vmax is not None else np.nanmax(data[mask])

    # Clip data to selected range
    data_clipped = np.clip(data, vmin, vmax)
    grid_clipped = griddata((lon, lat), data_clipped, (lon_mesh, lat_mesh), method='linear')

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()})
    cf = ax.contourf(lon_mesh, lat_mesh, grid_clipped, levels=100, cmap=cmap, vmin=vmin, vmax=vmax)
    cs = ax.contour(lon_mesh, lat_mesh, grid_clipped, levels=10, colors='black', linewidths=0.3)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.0f")

    if large:
        # High-resolution coastline for large display
        highres_coastline = NaturalEarthFeature(
            'physical', 'coastline', '10m',
            edgecolor='black', facecolor='none'
        )
        ax.add_feature(highres_coastline, linewidth=0.6)
    else:
        ax.coastlines()

    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.set_title(title, fontsize=14 if large else 12)
    cbar = fig.colorbar(cf, ax=ax, orientation='vertical', shrink=0.7)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(round(x))}"))

    return fig


# Plot results
if show_large:
    st.pyplot(plot_map(net_saving, "Energy savings from solar collector (kWh/day)", "jet", large=True))
else:
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)
    row3_col1, row3_col2 = st.columns(2)

    with row1_col1:
        st.pyplot(plot_map(net_saving, "Energy savings from solar collector (kWh/day)", "jet"))
    with row1_col2:
        st.pyplot(plot_map(helideck_gain, "Solar collector heating potential (kWh/day)", "jet"))
    with row2_col1:
        st.pyplot(plot_map(net_pool_heating, f"Net thermal energy required (kWh/day) to maintain {pool_temp}°C", "jet"))
    with row2_col2:
        st.pyplot(plot_map(pool_solar_gain, "Direct solar heating of pool (kWh/day)", "jet"))
    with row3_col1:
        st.pyplot(plot_map(ghi, "Global horizontal irradiation (kWh/day/m^2)", "jet"))
    with row3_col2:
        st.pyplot(plot_map(total_loss, "Total heat loss (kWh/day)", "jet"))