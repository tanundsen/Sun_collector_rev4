import numpy as np
from scipy.interpolate import interp1d

# Saturation pressure table from Appendix D
_temps_c = np.array([
    -40, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20,
    25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85
])
_sat_press_pa = np.array([
    12.84, 38, 63.25, 103.2, 165.2, 259.2, 401.5, 610.8, 871.9, 1227, 1704,
    2337, 3167, 4243, 5623, 7378, 9585, 12339, 14745, 19925, 25014, 31167,
    38554, 47365, 57809
])
_sat_pressure_fn = interp1d(_temps_c, _sat_press_pa, bounds_error=False, fill_value="extrapolate")

def saturation_pressure(temp_c):
    """Returns saturation vapor pressure in Pa for a given temperature in °C."""
    return _sat_pressure_fn(temp_c)

def compute_heat_losses(pool_temp, pool_area, pool_depth, T_day, T_night,
                        wind_day, wind_night, rh_day, rh_night,
                        night_hours, cover_used):
    """
    Compute total heat loss in kWh/day based on detailed model with evaporation, radiation, and convection.
    Returns Q_day and Q_night in kWh/day and breakdown of loss components.
    """
    seconds_per_hour = 3600
    hours_day = 24 - night_hours

    evap_fact = 1 #Evaporation tuning factor

    # Constants for radiation
    epsilon = 0.9  # emissivity
    sigma = 5.67e-8  # Stefan-Boltzmann constant W/m^2/K^4

    # Sky temperature estimation (5°C below air temperature)
    T_sky_day = T_day - 5
    T_sky_night = T_night - 5

    # Convert to Kelvin
    T_pool_K = pool_temp + 273.15
    T_sky_day_K = T_sky_day + 273.15
    T_sky_night_K = T_sky_night + 273.15

    # Evaporation heat loss (kW/m²)
    Pw_day = saturation_pressure(pool_temp)
    Pa_day = saturation_pressure(T_day) * rh_day / 100
    q_evap_day = evap_fact*((30.6 + 32.1 * wind_day) * (Pw_day - Pa_day)) / (3600 * 133.322)

    Pw_night = saturation_pressure(pool_temp)
    Pa_night = saturation_pressure(T_night) * rh_night / 100
    q_evap_night = evap_fact*((30.6 + 32.1 * wind_night) * (Pw_night - Pa_night)) / (3600 * 133.322)

    # Radiation loss (W/m²) converted to kW/m²
    q_rad_day = epsilon * sigma * (T_pool_K**4 - T_sky_day_K**4) / 1000
    q_rad_night = epsilon * sigma * (T_pool_K**4 - T_sky_night_K**4) / 1000

    # Convection loss using Ruiz and Martínez (2010) (kW/m²K)
    h_conv_day = (3.1 + 4.1 * wind_day) / 1000
    h_conv_night = (3.1 + 4.1 * wind_night) / 1000
    q_conv_day = h_conv_day * (pool_temp - T_day)
    q_conv_night = h_conv_night * (pool_temp - T_night)

    # If pool cover is used at night, reduce evaporation and radiation losses by 70 %
    if cover_used:
        q_evap_night *= 0.3
        q_rad_night *= 0.3

    # Total per m²
    q_total_day = q_evap_day + q_rad_day + q_conv_day
    q_total_night = q_evap_night + q_rad_night + q_conv_night

    # Total heat loss (kWh/day)
    Q_day = q_total_day * pool_area * hours_day
    Q_night = q_total_night * pool_area * night_hours

    Q_day = np.maximum(Q_day, 0)
    Q_night = np.maximum(Q_night, 0)

    # Component heat losses (kWh/day)
    evap_day = np.maximum(q_evap_day * pool_area * hours_day, 0)
    evap_night = np.maximum(q_evap_night * pool_area * night_hours, 0)
    rad_day = np.maximum(q_rad_day * pool_area * hours_day, 0)
    rad_night = np.maximum(q_rad_night * pool_area * night_hours, 0)
    conv_day = np.maximum(q_conv_day * pool_area * hours_day, 0)
    conv_night = np.maximum(q_conv_night * pool_area * night_hours, 0)

    return {
        "Q_day": Q_day,
        "Q_night": Q_night,
        "evap_day": evap_day,
        "evap_night": evap_night,
        "rad_day": rad_day,
        "rad_night": rad_night,
        "conv_day": conv_day,
        "conv_night": conv_night,
        "seconds_per_hour": seconds_per_hour,
        "hours_day": hours_day,
        "pool_volume": pool_area * pool_depth
    }
