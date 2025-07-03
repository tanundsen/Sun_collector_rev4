
from heat_loss_utils import compute_heat_losses
import streamlit as st
import pandas as pd
import numpy as np
from streamlit_folium import st_folium
import folium
import matplotlib.pyplot as plt

st.set_page_config(page_title="📍 Monthly Ship Location & Energy Savings", layout="wide")
st.title("📍 Monthly Ship Location & Energy Savings")

@st.cache_data
def load_data():
    return pd.read_csv("climate_data_sea.csv")

df = load_data()

# Sidebar input

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
    collector = plt.Circle((0, 0), collector_radius, color='yellow', alpha=1.0, label="Solar Collector")
    ax.add_patch(collector)

    # Draw helideck (black outline)
    helideck = plt.Circle((0, 0), helideck_radius, color='black', fill=False, linewidth=2, label="Helideck")
    ax.add_patch(helideck)

    # Labels
    ax.text(0, 0, "Solar\nCollector", color="black", fontsize=8, ha='center', va='center', weight='bold')
    ax.text(0, helideck_radius * 1.05, "Helideck", color="black", fontsize=8, ha='center', va='bottom', weight='bold')

    ax.set_xlim(-helideck_radius * 1.2, helideck_radius * 1.2)
    ax.set_ylim(-helideck_radius * 1.2, helideck_radius * 1.2)
    ax.set_aspect('equal')
    ax.axis('off')

    return fig

# Show in sidebar
st.sidebar.pyplot(plot_helideck_sidebar(helideck_diameter, collector_area))



cop = st.sidebar.slider("COP of Electric Heating System", 1.0, 6.0, 3.0)
usd_per_liter = st.sidebar.slider("Diesel Cost (USD/liter)", 0.5, 2.0, 1.2)
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
        "Highly shielded (5%) – partly enclosed"
    ],
    index=1  # Default is the second option
)
shielding_factor = {
    "Open exposure (70%) – e.g. Open deck without any obstructions": 0.7,
    "Partly shielded (40%) – some walls or windbreaks": 0.4,
    "Recessed or surrounded (15%) – Recessed or large wind breaks": 0.15,
    "Highly shielded (5%) – partly enclosed": 0.05
}[shielding]

months_ordered = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
short_months = [m[:3].lower() for m in months_ordered]

# Session state initialization
if "coords_by_month" not in st.session_state:
    st.session_state.coords_by_month = {}

# Month label mapping
month_display_labels = []
month_label_to_real = {}
for month in months_ordered:
    short = month[:3].lower()
    has_coords = short in st.session_state.coords_by_month
    label = f"✅ {month}" if has_coords else f"⬜ {month}"
    month_display_labels.append(label)
    month_label_to_real[label] = month

# Pick the next unassigned month by default
remaining_months = [m for m in months_ordered if m[:3].lower() not in st.session_state.coords_by_month]
default_label = f"⬜ {remaining_months[0]}" if remaining_months else month_display_labels[0]

selected_labels = st.multiselect(
    "Select month(s) to assign vessel location:",
    options=month_display_labels,
    default=[default_label],
    key="month_selection"
)
selected_months = [month_label_to_real[label] for label in selected_labels]

# Map section
st.markdown("Click a point on the map to set the ship's location for the selected month(s):")
m = folium.Map(location=[20, 0], zoom_start=2)
for mkey, coord in st.session_state.coords_by_month.items():
    folium.Marker(
        coord,
        tooltip=mkey.title(),
        popup=mkey.title(),
        icon=folium.DivIcon(html=f"<div style='font-size: 10pt'>📍 {mkey.title()}</div>")
    ).add_to(m)

map_data = st_folium(m, height=720, width=1200, returned_objects=["last_clicked"])

if selected_months and map_data and map_data.get("last_clicked"):
    latlng = (map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"])
    for m in selected_months:
        st.session_state.coords_by_month[m[:3].lower()] = latlng
    st.success(f"Saved location for {', '.join(selected_months)}: {latlng}")
    st.rerun()

# Show results only if all 12 months have a location
st.markdown("---")
st.subheader("📊 Monthly Summary of Energy Use and Savings")

if len(st.session_state.coords_by_month) == 12:
    results = []
    days_in_month = {
        "January": 31, "February": 28, "March": 31, "April": 30, "May": 31, "June": 30,
        "July": 31, "August": 31, "September": 30, "October": 31, "November": 30, "December": 31
    }
    hours_day = 24 - night_hours

    for month in months_ordered:
        short = month[:3].lower()
        lat_sel, lon_sel = st.session_state.coords_by_month[short]
        df["dist"] = np.sqrt((df["lat"] - lat_sel)**2 + (df["lon"] - lon_sel)**2)
        row = df.loc[df["dist"].idxmin()]

        T_min = row[f"tmin_{month}"]
        T_max = row[f"tmax_{month}"]
        T_avg = row.get(f"tavg_{month}", (T_min + T_max)/2)
        T_day = (T_avg + T_max) / 2
        T_night = (T_avg + T_min) / 2
        ghi = row[f"ghi_{month}"]
        wind = row[f"ws10m_{month}"]
        wind_day = wind * shielding_factor
        wind_night = 0.8 * wind * shielding_factor
        rh = row[f"rh_{month}"]
        rh_day = rh
        rh_night = 1.1 * rh

        loss = compute_heat_losses(pool_temp, pool_area, pool_depth, T_day, T_night,
                                   wind_day, wind_night, rh_day, rh_night, night_hours, cover_used)

        Q_day = loss["Q_day"]
        Q_night = loss["Q_night"]
        total_loss = Q_day + Q_night
        days = days_in_month[month]

        helideck_gain = ghi * helideck_area * collector_efficiency
        pool_solar_gain = ghi * pool_area * 0.7
        net_pool_heating = max(total_loss - pool_solar_gain, 0)
        net_saving = min(helideck_gain, net_pool_heating)

        electrical_saving = net_saving * days / cop
        diesel_kg = electrical_saving * 0.2
        el_energy_requierd = net_pool_heating / cop
        diesel_liters = diesel_kg / 0.84

        results.append({
            "Month": month,
            "Lat": round(lat_sel, 2),
            "Lon": round(lon_sel, 2),
            "Heat Loss (kWh)": round(total_loss * days, 1),
            "Pool direct solar heating (kWh)": round(pool_solar_gain * days, 1),
            "Net heat requirement (kWh)": round(net_pool_heating * days, 1),        
            "Electric heat requirement (kWh)": round(el_energy_requierd * days, 1),   
            "Solar collector thermal savings (kWh)": round(net_saving * days, 1),
            "Solar collector electric savings (kWh)": round(electrical_saving * days, 1),
            "Diesel Saved (liters)": round(diesel_liters, 1),
            "USD Saved": round(diesel_liters * usd_per_liter, 1),
        })
        
    df_result = pd.DataFrame(results)
    totals = df_result[[
        "Heat Loss (kWh)", "Pool direct solar heating (kWh)", "Net heat requirement (kWh)",
        "Electric heat requirement (kWh)", "Solar collector thermal savings (kWh)","Solar collector electric savings (kWh)","Diesel Saved (liters)","USD Saved"
    ]].sum()

    df_result.loc[len(df_result)] = {
        "Month": "Total", "Lat": "-", "Lon": "-",
        "Heat Loss (kWh)": round(totals["Heat Loss (kWh)"], 1),
        "Pool direct solar heating (kWh)": round(totals["Pool direct solar heating (kWh)"], 1),
        "Net heat requirement (kWh)": round(totals["Net heat requirement (kWh)"], 1),
        "Electric heat requirement (kWh)": round(totals["Electric heat requirement (kWh)"], 1),
        "Solar collector thermal savings (kWh)": round(totals["Solar collector thermal savings (kWh)"], 1),
        "Solar collector electric savings (kWh)": round(totals["Solar collector electric savings (kWh)"], 1),
        "Diesel Saved (liters)": round(totals["Diesel Saved (liters)"], 1),
        "USD Saved": round(totals["USD Saved"], 1),
    }

    st.dataframe(df_result.set_index("Month"), use_container_width=True, height=(df_result.shape[0] + 1) * 35)

    # Plot: Monthly Energy
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    x = df_result['Month'][:-1]
    loss = df_result['Net heat requirement (kWh)'].iloc[:-1]
    saving = df_result['Solar collector thermal savings (kWh)'].iloc[:-1]
    ax1.bar(x, loss, label='Energy Needed', color='lightgray')
    ax1.bar(x, saving, label='Net Saving', color='green')
    ax1.set_ylabel('Energy (kWh/month)')
    ax1.set_title('Monthly Energy Use vs. Solar Savings')
    ax1.legend()
    plt.xticks(rotation=45)  # ← This fixes the overlap
    st.pyplot(fig1)

    # Plot: USD Savings
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    usd_saved = df_result['USD Saved'].iloc[:-1]
    ax2.bar(x, usd_saved, color='skyblue')
    ax2.set_ylabel('USD Saved')
    ax2.set_title('Monthly Diesel Cost Savings')
    plt.xticks(rotation=45)  # ← Add here too
    st.pyplot(fig2)

