import streamlit as st
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
import tempfile
from geopy.geocoders import Nominatim
import time
import gc
from threading import Lock
import warnings
warnings.filterwarnings("ignore")

# Atmospheric science imports
try:
    from netCDF4 import Dataset, num2date
    from metpy import calc as mpcalc
    from metpy.units import units
    from metpy.plots import SkewT, Hodograph
    import cdsapi
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    st.error(f"Missing dependencies: {e}")

# Set page config
st.set_page_config(
    page_title="ERA5 Sounding Plotter", 
    page_icon="🌪️",
    layout="wide"
)

# Resource management
processing_lock = Lock()

def check_dependencies():
    """Check if all required dependencies are available"""
    missing = []
    try:
        import netCDF4
    except ImportError:
        missing.append("netCDF4")
    
    try:
        import metpy
    except ImportError:
        missing.append("metpy")
        
    try:
        import cdsapi
    except ImportError:
        missing.append("cdsapi")
    
    return len(missing) == 0, missing

def get_coordinates(location_str):
    """Geocoding with timeout"""
    try:
        geolocator = Nominatim(user_agent="era5_sounding_viewer", timeout=5)
        time.sleep(1)
        location = geolocator.geocode(location_str, timeout=5)
        if location:
            return location.latitude, location.longitude
        return None, None
    except Exception as e:
        st.error(f"Geocoding error: {str(e)}")
        return None, None

def download_era5_data(year, month, day, hour, lat, lon, progress_bar):
    """Download ERA5 data from CDS"""
    progress_bar.progress(10, "Connecting to Copernicus Climate Data Store...")
    
    try:
        c = cdsapi.Client()
        
        # Small area around the point
        area = [lat + 0.01, lon - 0.01, lat - 0.01, lon + 0.01]
        
        dataset = "reanalysis-era5-pressure-levels"
        request = {
            "product_type": ["reanalysis"],
            "variable": [
                "geopotential",
                "relative_humidity",
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind"
            ],
            "pressure_level": [
                "1", "2", "3", "5", "7", "10", "20", "30", "50",
                "70", "100", "125", "150", "175", "200", "225",
                "250", "300", "350", "400", "450", "500", "550",
                "600", "650", "700", "750", "775", "800", "825",
                "850", "875", "900", "925", "950", "975", "1000"
            ],
            "year": [year],           
            "month": [month],         
            "day": [day],             
            "time": [f"{hour:02d}:00"],
            "area": area,
            "format": "netcdf"
        }
        
        # Create temporary file using original naming pattern
        file_path = f"era5_{year}{month:02d}{day:02d}_{hour:02d}.nc"
        temp_path = os.path.join(tempfile.gettempdir(), file_path)
        
        progress_bar.progress(25, "Submitting ERA5 data request...")
        
        # Submit request using original method
        c.retrieve(dataset, request, temp_path)
        
        progress_bar.progress(75, "ERA5 data downloaded successfully!")
        
        return temp_path
        
    except Exception as e:
        raise Exception(f"ERA5 download failed: {str(e)}")

def process_era5_dataset(file_path, lat, lon, progress_bar):
    """Process ERA5 netCDF file to extract sounding data - IDENTICAL TO ORIGINAL"""
    progress_bar.progress(80, "Processing atmospheric data...")
    
    try:
        ds = Dataset(file_path)
        
        # Find nearest grid point - EXACT ORIGINAL LOGIC
        latitudes = ds['latitude'][:]
        longitudes = ds['longitude'][:]
        x, y = np.abs(latitudes - lat).argmin(), np.abs(longitudes - lon).argmin()
        
        # Extract variables - EXACT ORIGINAL LOGIC
        levels = ds['pressure_level'][:]
        z = ds['z'][0, :, y, x] / 9.81  # Convert geopotential to height
        u = ds['u'][0, :, y, x]
        v = ds['v'][0, :, y, x]
        temp = ds['t'][0, :, y, x] - 273.15  # Convert to Celsius
        rh = ds['r'][0, :, y, x]
        
        # Calculate derived variables - EXACT ORIGINAL LOGIC
        wdir = np.degrees(np.arctan2(u, v)) + 180
        wspd = np.hypot(u, v) * 1.944  # Convert to knots
        
        # Calculate dewpoint - EXACT ORIGINAL LOGIC
        dew = np.asarray(
            mpcalc.dewpoint_from_relative_humidity(
                np.asarray(temp) * units.degC, 
                np.asarray(rh) * units.percent
            )
        )
        
        # Get valid time - EXACT ORIGINAL LOGIC
        valid_time = ds['valid_time'][:]
        valid_date = num2date(valid_time[0], ds['valid_time'].units)
        
        ds.close()
        
        return {
            "valid_date": valid_date,
            "levels": levels,
            "z": z,
            "u": u,
            "v": v,
            "temp": temp,
            "dew": dew,
            "wdir": wdir,
            "wspd": wspd
        }
        
    except Exception as e:
        raise Exception(f"Data processing failed: {str(e)}")

def create_comprehensive_sounding_plot(data, lat, progress_bar):
    """Create comprehensive sounding plot identical to original and save as JPG"""
    progress_bar.progress(85, "Creating comprehensive sounding plot...")
    
    try:
        # Create temporary file path for the plot
        plot_filename = f"era5_sounding_{int(time.time())}.jpg"
        temp_plot_path = os.path.join(tempfile.gettempdir(), plot_filename)
        
        # Use the ORIGINAL comprehensive plotting function
        create_plotB_comprehensive(data, temp_plot_path, lat)
        
        # Read the image file and return as matplotlib figure for Streamlit
        from PIL import Image
        img = Image.open(temp_plot_path)
        
        # Convert PIL image to matplotlib figure
        fig, ax = plt.subplots(figsize=(18, 12))
        ax.imshow(img)
        ax.axis('off')
        
        # Clean up temp file
        if os.path.exists(temp_plot_path):
            os.remove(temp_plot_path)
            
        return fig
        
    except Exception as e:
        raise Exception(f"Comprehensive plot creation failed: {str(e)}")

def create_plotB_comprehensive(data, file_path, lat=None):
    """EXACT COPY of create_plotB from original script - creates identical plot"""
    import matplotlib.gridspec as gridspec

    try:
        # Convert ERA5 data to the format our plotting logic expects
        pressure = np.array(data['levels'])
        height = np.array(data['z'])
        temperature = np.array(data['temp'])
        dewpoint = np.array(data['dew'])
        u_wind = np.array(data['u']) * 1.944  # Convert m/s to knots
        v_wind = np.array(data['v']) * 1.944  # Convert m/s to knots
        wspd = np.array(data['wspd'])
        wdir = np.array(data['wdir'])

        # Sort by decreasing pressure (surface to top)
        sort_idx = np.argsort(pressure)[::-1]
        pressure = pressure[sort_idx]
        height = height[sort_idx]
        temperature = temperature[sort_idx]
        dewpoint = dewpoint[sort_idx]
        u_wind = u_wind[sort_idx]
        v_wind = v_wind[sort_idx]
        wspd = wspd[sort_idx]
        wdir = wdir[sort_idx]

        # Filter valid data
        valid_mask = (
                (pressure > 0) &
                (np.isfinite(pressure)) & (np.isfinite(temperature)) & (np.isfinite(dewpoint)) &
                (temperature > -100) & (temperature < 60) &
                (dewpoint > -100) & (dewpoint < 60) &
                (dewpoint <= temperature + 0.1)
        )

        # Apply mask
        p = pressure[valid_mask] * units.hPa
        T = temperature[valid_mask] * units.degC
        Td = dewpoint[valid_mask] * units.degC
        height_m = height[valid_mask] * units.meter
        u = u_wind[valid_mask] * units.knot
        v = v_wind[valid_mask] * units.knot

        # Calculate wind components for hodograph
        wdir_rad = np.deg2rad(wdir[valid_mask])
        u_calc = -wspd[valid_mask] * np.sin(wdir_rad) * units.knot
        v_calc = -wspd[valid_mask] * np.cos(wdir_rad) * units.knot

        # Determine hemisphere - CRITICAL FIX
        is_southern_hemisphere = lat is not None and lat < 0

        # Calculate basic parameters with error handling
        try:
            # Surface-based parcel
            sb_parcel_prof = mpcalc.parcel_profile(p, T[0], Td[0])
            sb_cape, sb_cin = mpcalc.cape_cin(p, T, Td, sb_parcel_prof)

            # Mixed layer parcel
            ml_t, ml_td = mpcalc.mixed_layer(p, T, Td, depth=50 * units.hPa)
            ml_parcel_prof = mpcalc.parcel_profile(p, ml_t, ml_td)
            ml_cape, ml_cin = mpcalc.mixed_layer_cape_cin(p, T, ml_parcel_prof, depth=50 * units.hPa)

            # Most unstable parcel
            mu_cape, mu_cin = mpcalc.most_unstable_cape_cin(p, T, Td, depth=50 * units.hPa)

            # LCL
            lcl_p, lcl_t = mpcalc.lcl(p[0], T[0], Td[0])

            # LCL height estimation
            new_p = np.append(p[p > lcl_p], lcl_p)
            new_t = np.append(T[p > lcl_p], lcl_t)
            lcl_height = mpcalc.thickness_hydrostatic(new_p, new_t)
        except:
            sb_cape = ml_cape = mu_cape = 0 * units('J/kg')
            sb_cin = ml_cin = mu_cin = 0 * units('J/kg')
            lcl_p, lcl_t = p[0], T[0]
            lcl_height = 1000 * units.meter

        try:
            # Storm motion - both RM and LM
            storm_motion = mpcalc.bunkers_storm_motion(p, u_calc, v_calc, height_m)
            storm_u_rm, storm_v_rm = storm_motion[0]
            storm_u_lm, storm_v_lm = storm_motion[1]

            # HEMISPHERE-SPECIFIC STORM MOTION SELECTION
            if is_southern_hemisphere:
                # In Southern Hemisphere, use LM storm motion as primary
                primary_storm_u, primary_storm_v = storm_u_lm, storm_v_lm
                storm_type_label = "LM"
                line_color = 'red'  # LM color
            else:
                # In Northern Hemisphere, use RM storm motion as primary
                primary_storm_u, primary_storm_v = storm_u_rm, storm_v_rm
                storm_type_label = "RM"
                line_color = 'green'  # RM color

            # Shear calculations
            shear_1km = mpcalc.wind_speed(*mpcalc.bulk_shear(p, u_calc, v_calc, height=height_m, depth=1 * units.km))
            shear_3km = mpcalc.wind_speed(*mpcalc.bulk_shear(p, u_calc, v_calc, height=height_m, depth=3 * units.km))
            shear_6km = mpcalc.wind_speed(*mpcalc.bulk_shear(p, u_calc, v_calc, height=height_m, depth=6 * units.km))
            shear_8km = mpcalc.wind_speed(*mpcalc.bulk_shear(p, u_calc, v_calc, height=height_m, depth=8 * units.km))

            # SRH calculations - HEMISPHERE-SPECIFIC
            if is_southern_hemisphere:
                # Use LM storm motion for SRH calculations in SHEM
                srh_1km = mpcalc.storm_relative_helicity(height_m, u_calc, v_calc, depth=1 * units.km,
                                                         storm_u=storm_u_lm, storm_v=storm_v_lm)[2]
                srh_3km = mpcalc.storm_relative_helicity(height_m, u_calc, v_calc, depth=3 * units.km,
                                                         storm_u=storm_u_lm, storm_v=storm_v_lm)[2]
            else:
                # Use RM storm motion for SRH calculations in NHEM
                srh_1km = mpcalc.storm_relative_helicity(height_m, u_calc, v_calc, depth=1 * units.km,
                                                         storm_u=storm_u_rm, storm_v=storm_v_rm)[2]
                srh_3km = mpcalc.storm_relative_helicity(height_m, u_calc, v_calc, depth=3 * units.km,
                                                         storm_u=storm_u_rm, storm_v=storm_v_rm)[2]

            # Effective layer calculations
            effective_bottom, effective_top = 0, len(p) // 2
            try:
                for i in range(len(p)):
                    parcel_prof = mpcalc.parcel_profile(p[i:], T[i], Td[i])
                    cape, cin = mpcalc.cape_cin(p[i:], T[i:], Td[i:], parcel_prof)
                    if cape >= 100 * units('J/kg') and cin >= -250 * units('J/kg'):
                        if effective_bottom == 0:
                            effective_bottom = i
                        effective_top = i

                # Effective SRH and bulk shear - HEMISPHERE-SPECIFIC
                if effective_bottom < len(height_m) and effective_top < len(height_m):
                    eff_depth = height_m[effective_top] - height_m[effective_bottom]

                    # Use hemisphere-appropriate storm motion for effective SRH
                    esrh = mpcalc.storm_relative_helicity(
                        height_m[effective_bottom:effective_top + 1],
                        u_calc[effective_bottom:effective_top + 1],
                        v_calc[effective_bottom:effective_top + 1],
                        depth=eff_depth,
                        storm_u=primary_storm_u,
                        storm_v=primary_storm_v
                    )[2]

                    ebwd_u, ebwd_v = mpcalc.bulk_shear(
                        p[effective_bottom:effective_top + 1],
                        u_calc[effective_bottom:effective_top + 1],
                        v_calc[effective_bottom:effective_top + 1],
                        height=height_m[effective_bottom:effective_top + 1]
                    )
                    ebwd = mpcalc.wind_speed(ebwd_u, ebwd_v)
                else:
                    esrh = srh_3km
                    ebwd = shear_6km
            except:
                esrh = srh_3km
                ebwd = shear_6km

            # Calculate STP parameters using hemisphere-appropriate SRH values
            try:
                # STPF (Thompson et al. 2003)
                stpf = (sb_cape / (1500 * units('J/kg'))) * \
                       (srh_1km / (150 * units('m^2/s^2'))) * \
                       (shear_6km / (20 * units('m/s'))) * \
                       ((2000 * units.meter - lcl_height) / (1000 * units.meter))
                stpf = stpf.to('dimensionless').magnitude

                # STPE (Thompson et al. 2012)
                stpe = (ml_cape / (1500 * units('J/kg'))) * \
                       (esrh / (150 * units('m^2/s^2'))) * \
                       (ebwd / (12 * units('m/s'))) * \
                       ((2000 * units.meter - lcl_height) / (1000 * units.meter)) * \
                       ((ml_cin + 200 * units('J/kg')) / (150 * units('J/kg')))
                stpe = stpe.to('dimensionless').magnitude
            except:
                stpf = stpe = 0

            # Critical angle calculation using hemisphere-appropriate storm motion
            try:
                critical_angle = mpcalc.critical_angle(p, u_calc, v_calc, height_m, primary_storm_u, primary_storm_v)
            except:
                critical_angle = 0 * units.degrees

        except:
            storm_u_rm = storm_v_rm = storm_u_lm = storm_v_lm = 0 * units.knot
            primary_storm_u = primary_storm_v = 0 * units.knot
            shear_1km = shear_3km = shear_6km = shear_8km = 0 * units('m/s')
            srh_1km = srh_3km = esrh = 0 * units('m^2/s^2')
            ebwd = 0 * units('m/s')
            stpf = stpe = 0
            critical_angle = 0 * units.degrees
            storm_type_label = "RM"
            line_color = 'green'

        # Calculate 3CAPE - CAPE within the lowest 3 kilometers only
        try:
            # Find the 3km level above ground
            surface_height = height_m[0]
            km3_height = surface_height + 3000 * units.meter

            # Find indices for levels within the lowest 3km
            km3_mask = height_m <= km3_height

            if np.any(km3_mask):
                # Extract data for lowest 3km only
                p_3km = p[km3_mask]
                T_3km = T[km3_mask]
                Td_3km = Td[km3_mask]

                # Calculate parcel profile for surface-based parcel within 3km
                sb_parcel_3km = mpcalc.parcel_profile(p_3km, T[0], Td[0])
                cape_3km, _ = mpcalc.cape_cin(p_3km, T_3km, Td_3km, sb_parcel_3km)
            else:
                cape_3km = 0 * units('J/kg')
        except:
            cape_3km = 0 * units('J/kg')

        # Simple lapse rate calculations
        try:
            sfc_height = height_m[0]
            # Find 1km, 3km levels
            km1_idx = np.argmin(np.abs(height_m - (sfc_height + 1000 * units.meter)))
            km3_idx = np.argmin(np.abs(height_m - (sfc_height + 3000 * units.meter)))

            lapse_1km = (T[0] - T[km1_idx]) / (1 * units.km)
            lapse_3km = (T[0] - T[km3_idx]) / (3 * units.km)
        except:
            lapse_1km = lapse_3km = 6.5 * units('K/km')

        # === PLOTTING SECTION - DARK THEME ===

        # Set up the figure with dark theme
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(18, 12), facecolor='#2F2F2F')
        gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[2.5, 1])

        # TOP LEFT: Skew-T plot with dark theme
        skew = SkewT(fig, rotation=45, subplot=gs[0, 0])
        skew.ax.set_facecolor('#2F2F2F')

        # Plot the data with dark theme colors
        skew.plot(p, T, 'r', linewidth=2.5)
        skew.plot(p, Td, 'g', linewidth=2.5)
        skew.plot_barbs(p, u_calc, v_calc, barbcolor='white', flagcolor='white')

        # Set limits
        skew.ax.set_ylim(1000, 100)
        skew.ax.set_xlim(-40, 60)

        # Labels with white text
        skew.ax.set_xlabel(f'Temperature ({T.units:~P})', color='white')
        skew.ax.set_ylabel(f'Pressure ({p.units:~P})', color='white')
        skew.ax.tick_params(colors='white')

        # Plot LCL as white dot
        skew.plot(lcl_p, lcl_t, 'wo', markerfacecolor='white', markeredgecolor='black', markersize=8)

        # Calculate and plot parcel profile
        try:
            prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')
            skew.plot(p, prof, color='#FFFF99', linewidth=2, linestyle='--')  # Light yellow parcel path
            skew.shade_cin(p, T, prof, Td, alpha=0.4)
            skew.shade_cape(p, T, prof, alpha=0.4)
        except:
            pass

        # Add reference lines in white
        skew.ax.axvline(0, color='white', linestyle='--', linewidth=2, alpha=0.7)
        skew.plot_dry_adiabats(colors='white', alpha=0.3)
        skew.plot_moist_adiabats(colors='white', alpha=0.3)
        skew.plot_mixing_lines(colors='white', alpha=0.3)

        # Title and grid with white text
        skew.ax.set_title('Skew-T Log-P Diagram', fontsize=14, fontweight='bold', color='white')
        skew.ax.grid(True, alpha=0.3, color='white')

        # TOP RIGHT: Hodograph with dark theme
        hodo_ax = fig.add_subplot(gs[0, 1], facecolor='#2F2F2F')
        h = Hodograph(hodo_ax, component_range=80)
        h.add_grid(increment=20, color='white', alpha=0.3)

        # Color-code hodograph by height layers
        try:
            height_agl = height_m - height_m[0]
            layers = [
                (0, 3000, 'red', '0-3km'),
                (3000, 6000, 'green', '3-6km'),
                (6000, 9000, 'yellow', '6-9km'),
                (9000, 20000, 'lightblue', '>9km')
            ]

            for bottom, top, color, label in layers:
                height_values = height_agl.to('meter').magnitude
                mask = (height_values >= bottom) & (height_values < top)
                if np.any(mask):
                    h.plot(u_calc[mask], v_calc[mask], color=color, linewidth=3, label=label)

            # Add both RM and LM storm motion vectors
            h.ax.plot(storm_u_rm.magnitude, storm_v_rm.magnitude, 'go', markersize=8,
                      markerfacecolor='green', markeredgecolor='white', linewidth=2, label='RM Storm Motion')
            h.ax.plot(storm_u_lm.magnitude, storm_v_lm.magnitude, 'ro', markersize=8,
                      markerfacecolor='red', markeredgecolor='white', linewidth=2, label='LM Storm Motion')

            # CRITICAL FIX: Add critical angle line to hemisphere-appropriate storm motion
            h.ax.plot([0, primary_storm_u.magnitude], [0, primary_storm_v.magnitude],
                      '--', color=line_color, alpha=0.8, linewidth=2)
        except:
            pass

        # Update hodograph title to show which storm motion is primary
        hemisphere_text = "SHEM" if is_southern_hemisphere else "NHEM"
        hodo_ax.set_title(f'Hodograph ({hemisphere_text} - {storm_type_label} Primary)',
                          fontsize=14, fontweight='bold', color='white')
        hodo_ax.tick_params(colors='white')

        # Add critical angle text
        try:
            angle_value = critical_angle.magnitude
            degree_text = "degrees" if angle_value != 1 else "degree"
            hodo_ax.text(0.02, 0.02, f'Critical Angle ({storm_type_label}): {angle_value:.0f} {degree_text}',
                         transform=hodo_ax.transAxes, fontsize=11, fontweight='bold',
                         color='white', bbox=dict(boxstyle="round,pad=0.3",
                                                  facecolor='#2F2F2F', alpha=0.9, edgecolor='white'))
        except:
            pass

        # BOTTOM LEFT: All Parameters with dark theme
        params_ax = fig.add_subplot(gs[1, 0], facecolor='#2F2F2F')
        params_ax.axis('off')

        # Calculate base positions with offset
        base_col1 = 0.02 + 0.125
        base_col2 = 0.30 + 0.125
        base_col3 = 0.58 + 0.125

        # Organize parameters in columns - UPDATE STORM MOTION LABELS
        thermodynamic_params = [
            f"SBCAPE: {sb_cape:~.0f}",
            f"MLCAPE: {ml_cape:~.0f}",
            f"MUCAPE: {mu_cape:~.0f}",
            f"SBCIN: {sb_cin:~.0f}",
            f"MLCIN: {ml_cin:~.0f}",
            f"3CAPE: {cape_3km:~.0f}",
            f"LCL Height: {lcl_height:~.0f}"
        ]

        kinematic_params = [
            f"0-1km Shear: {shear_1km:~.1f}",
            f"0-3km Shear: {shear_3km:~.1f}",
            f"0-6km Shear: {shear_6km:~.1f}",
            f"0-8km Shear: {shear_8km:~.1f}",
            f"Eff Bulk Shear: {ebwd:~.1f}",
            f"0-1km SRH ({storm_type_label}): {srh_1km:~.0f}",
            f"0-3km SRH ({storm_type_label}): {srh_3km:~.0f}",
            f"Eff SRH ({storm_type_label}): {esrh:~.0f}"
        ]

        composite_params = [
            f"STPF: {stpf:.2f}",
            f"STPE: {stpe:.2f}",
            f"RM Storm U: {storm_u_rm:~.1f}",
            f"RM Storm V: {storm_v_rm:~.1f}",
            f"LM Storm U: {storm_u_lm:~.1f}",
            f"LM Storm V: {storm_v_lm:~.1f}",
            f"0-1km Lapse: {lapse_1km:~.1f}"
        ]

        # Titles with white text
        title_y = 0.95
        params_ax.text(base_col1, title_y, "THERMODYNAMIC PARAMETERS:",
                       transform=params_ax.transAxes, fontsize=11, fontweight='bold', fontfamily='monospace',
                       color='white')
        params_ax.text(base_col2, title_y, "KINEMATIC PARAMETERS:",
                       transform=params_ax.transAxes, fontsize=11, fontweight='bold', fontfamily='monospace',
                       color='white')
        params_ax.text(base_col3, title_y, "COMPOSITE PARAMETERS:",
                       transform=params_ax.transAxes, fontsize=11, fontweight='bold', fontfamily='monospace',
                       color='white')

        # Parameters with white text
        param_start_y = 0.85
        for i, text in enumerate(thermodynamic_params):
            params_ax.text(base_col1, param_start_y - i * 0.1, text, transform=params_ax.transAxes,
                           fontsize=10, fontfamily='monospace', color='white')

        for i, text in enumerate(kinematic_params):
            params_ax.text(base_col2, param_start_y - i * 0.1, text, transform=params_ax.transAxes,
                           fontsize=10, fontfamily='monospace', color='white')

        for i, text in enumerate(composite_params):
            params_ax.text(base_col3, param_start_y - i * 0.1, text, transform=params_ax.transAxes,
                           fontsize=10, fontfamily='monospace', color='white')

        params_ax.set_title('Calculated Parameters', fontweight='bold', fontsize=14, color='white')

        # BOTTOM RIGHT: Hazard Assessment with dark theme - USING ABSOLUTE VALUES
        hazard_ax = fig.add_subplot(gs[1, 1], facecolor='#2F2F2F')
        hazard_ax.axis('off')

        # Enhanced hazard assessment with absolute SRH values (CRITICAL FIX)
        try:
            cape_val = sb_cape.to('J/kg').magnitude
            ml_cape_val = ml_cape.to('J/kg').magnitude
            shear_6km_val = shear_6km.to('m/s').magnitude
            shear_8km_val = shear_8km.to('m/s').magnitude
            # CRITICAL: Use ABSOLUTE VALUE of hemisphere-appropriate SRH values
            srh_1km_val = abs(srh_1km.to('m^2/s^2').magnitude)
            srh_3km_val = abs(srh_3km.to('m^2/s^2').magnitude)
            esrh_val = abs(esrh.to('m^2/s^2').magnitude)
            lcl_height_val = lcl_height.to('m').magnitude

            # Enhanced logic matching the comprehensive analysis
            if (stpf >= 8 or stpe >= 10 or
                    (stpf >= 4 and stpe >= 6 and srh_3km_val >= 400 and shear_6km_val >= 25) or
                    (srh_3km_val >= 500 and shear_6km_val >= 30 and cape_val >= 2000)):
                hazard_type = "PDS TOR"
                hazard_color = "#C85A8A"
            elif (stpf >= 3 or stpe >= 4 or
                  (srh_3km_val >= 300 and shear_6km_val >= 20 and cape_val >= 1500) or
                  (stpf >= 2 and stpe >= 3 and srh_3km_val >= 200)):
                hazard_type = "TOR"
                hazard_color = "red"
            elif (stpf >= 1 or stpe >= 1.5 or
                  (srh_3km_val >= 150 and shear_6km_val >= 15 and cape_val >= 1000)):
                hazard_type = "MRGL TOR"
                hazard_color = "darkred"
            elif (shear_6km_val >= 15 and cape_val >= 1000):
                hazard_type = "SVR"
                hazard_color = "orange"
            elif (shear_6km_val >= 10 and cape_val >= 500):
                hazard_type = "MRGL SVR"
                hazard_color = "yellow"
            elif (cape_val >= 2500 and shear_6km_val < 10):
                hazard_type = "FLASH FLOOD"
                hazard_color = "blue"
            elif cape_val < 100:
                hazard_type = "NONE"
                hazard_color = "green"
            else:
                hazard_type = "SVR"
                hazard_color = "orange"
        except:
            hazard_type = "UNKNOWN"
            hazard_color = "gray"

        # Hazard box layout with white text
        hazard_ax.text(0.5, 0.80, "PREDICTED", transform=hazard_ax.transAxes,
                       fontsize=16, fontweight='bold', ha='center', color='white')
        hazard_ax.text(0.5, 0.65, "HAZARD TYPE", transform=hazard_ax.transAxes,
                       fontsize=16, fontweight='bold', ha='center', color='white')
        hazard_ax.text(0.5, 0.40, hazard_type, transform=hazard_ax.transAxes,
                       fontsize=22, fontweight='bold', ha='center', color=hazard_color,
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='#2F2F2F',
                                 edgecolor=hazard_color, linewidth=3))

        # Add title with white text and hemisphere info
        formatted_date = data['valid_date'].strftime("%Y-%m-%d %H:%M UTC")
        fig.suptitle(f'Comprehensive Sounding Analysis - ERA5 ({hemisphere_text}) - {formatted_date}',
                     fontsize=16, fontweight='bold', color='white')

        # Add attribution with white text
        plt.figtext(0.5, 0.02, 'Plotted by @Sekai_WX',
                    ha='center', fontsize=8, style='italic', color='white')

        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.90, left=0.00, right=0.95, hspace=0.3, wspace=0.2)

        # Save the plot as JPG with dark background
        plt.savefig(file_path, format='jpg', dpi=150, bbox_inches='tight', facecolor='#2F2F2F')
        plt.close()

    except Exception as e:
        print(f"Enhanced plotting error: {e}")
        # Fallback if anything fails
        raise Exception(f"Comprehensive plot creation failed: {str(e)}")

def process_era5_sounding(date_input, hour, lat, lon):
    """Main processing function for ERA5 sounding"""
    
    if not processing_lock.acquire(blocking=False):
        raise Exception("Another sounding request is processing. Please wait and try again.")
    
    try:
        year = date_input.year
        month = date_input.month
        day = date_input.day
        
        progress_bar = st.progress(0, "Initializing ERA5 sounding request...")
        
        # Download ERA5 data
        file_path = download_era5_data(year, month, day, hour, lat, lon, progress_bar)
        
        try:
            # Process data
            data = process_era5_dataset(file_path, lat, lon, progress_bar)
            
            # Create comprehensive plot
            fig = create_comprehensive_sounding_plot(data, lat, progress_bar)
            
            progress_bar.progress(100, "Complete!")
            time.sleep(0.5)
            progress_bar.empty()
            
            return fig, data
            
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
            gc.collect()
            
    finally:
        processing_lock.release()

# Set up CDS credentials using your API key
def setup_cds_credentials():
    """Set up CDS API credentials using developer's API key"""
    try:
        # Try to get from Streamlit secrets first (most secure)
        if hasattr(st, 'secrets') and 'cdsapi' in st.secrets:
            cds_key = st.secrets['cdsapi']['key']
        else:
            # Fallback to your hardcoded key (replace with your actual key)
            cds_key = "YOUR_CDS_API_KEY_HERE"  # Replace this with your actual key
            if cds_key == "YOUR_CDS_API_KEY_HERE":
                st.error("❌ CDS API key not configured. Please contact the developer.")
                return False
        
        os.environ['CDSAPI_URL'] = 'https://cds.climate.copernicus.eu/api'
        os.environ['CDSAPI_KEY'] = cds_key
        return True
        
    except Exception as e:
        st.error(f"❌ CDS API setup failed: {e}")
        return False

# Streamlit UI
st.title("🌪️ ERA5 Atmospheric Sounding Plotter")
st.markdown("### Comprehensive Atmospheric Profile Analysis")

if not DEPENDENCIES_AVAILABLE:
    st.error("❌ Required atmospheric science libraries are not installed.")
    st.info("This tool requires: netCDF4, metpy, and cdsapi")
    st.stop()

st.markdown("""
Generate **ERA-5 Reanalysis Sounding Plots** using this simple application. Just select your location, date, and hit the generate button!
""")

# Initialize CDS credentials
cds_setup_success = setup_cds_credentials()
if not cds_setup_success:
    st.stop()

col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("📅 Date & Time")
    
    # Date input (ERA5 is available with ~5 day delay)
    today = datetime.now().date()
    max_date = today - timedelta(days=5)  # ERA5 delay
    min_date = datetime(1940, 1, 1).date()  # ERA5 start
    
    date_input = st.date_input(
        "Date", 
        value=max_date,
        min_value=min_date, 
        max_value=max_date,
        help="ERA5 data available from 1940 to ~5 days ago"
    )
    
    hour_input = st.selectbox(
        "Hour (UTC)", 
        options=list(range(24)), 
        index=12,
        format_func=lambda x: f"{x:02d}:00",
        help="ERA5 provides hourly data (00-23 UTC)"
    )
    
    st.subheader("🌍 Location")
    
    location_method = st.radio(
        "Location Input Method",
        ["City/Place Name", "Coordinates (Lat, Lon)"]
    )
    
    if location_method == "City/Place Name":
        location_input = st.text_input(
            "Enter location", 
            placeholder="e.g., Sydney, Moore OK, Buenos Aires"
        )
        lat, lon = None, None
        if location_input:
            with st.spinner("Finding coordinates..."):
                lat, lon = get_coordinates(location_input)
                if lat and lon:
                    hemisphere = "Southern" if lat < 0 else "Northern"
                    st.success(f"📍 {lat:.4f}°, {lon:.4f}° ({hemisphere} Hemisphere)")
                else:
                    st.error("Location not found. Try coordinates instead.")
    else:
        col_lat, col_lon = st.columns(2)
        with col_lat:
            lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, 
                                value=-33.87, step=0.1)
        with col_lon:
            lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, 
                                value=151.21, step=0.1)
        
        if lat is not None:
            hemisphere = "Southern" if lat < 0 else "Northern"
            st.info(f"📍 {hemisphere} Hemisphere")
    
    # Warning about processing time
    st.info("⏱️ **Processing Time:** 1-3 minutes (ERA5 hourly data download)")
    
    generate_button = st.button("🚀 Generate Comprehensive Sounding Analysis", type="primary")

with col2:
    st.subheader("📊 Comprehensive Atmospheric Sounding")
    
    if generate_button:
        if lat is not None and lon is not None:
            try:
                fig, data = process_era5_sounding(date_input, hour_input, lat, lon)
                
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                gc.collect()
                
                st.success("✅ Comprehensive ERA5 sounding analysis complete!")
                st.info("💡 Right-click on the plot to save it to your device as JPG.")
                
                # Additional info
                valid_date = data['valid_date']
                st.caption(f"📅 Analysis time: {valid_date.strftime('%Y-%m-%d %H:%M UTC')}")
                
            except Exception as e:
                st.error(f"❌ Error generating sounding: {str(e)}")
                if "CDS" in str(e) or "API" in str(e):
                    st.info("💡 Check your CDS API key and internet connection.")
                elif "Another request" in str(e):
                    st.info("💡 Only one user can process at a time. Please wait and retry.")
        else:
            st.warning("⚠️ Please provide a valid location.")

# Information sections
with st.expander("🌪️ About ERA5 Atmospheric Soundings"):
    st.markdown("""
    **ERA5** is ECMWF's fifth-generation atmospheric reanalysis providing comprehensive 
    atmospheric data from 1940 to near real-time.
    
    **Comprehensive Sounding Analysis Features:**
    - **Skew-T log-P diagrams** with temperature, dewpoint, and wind profiles
    - **Hodographs** showing wind patterns by altitude with storm motion vectors
    - **CAPE/CIN calculations** for convective potential assessment
    - **Wind shear analysis** at multiple levels (0-1km, 0-3km, 0-6km, 0-8km)
    - **Storm-Relative Helicity (SRH)** with hemisphere-specific calculations
    - **Supercell Tornado Parameter (STP)** for severe weather potential
    - **Automated hazard assessment** (TOR, SVR, MRGL, etc.)
    - **Dark professional theme** matching NCEP operational products
    
    **Hemisphere-Specific Features:**
    - **Northern Hemisphere**: Uses Right-Moving (RM) storm motion as primary
    - **Southern Hemisphere**: Uses Left-Moving (LM) storm motion as primary
    - **Automatic detection** based on latitude
    """)

with st.expander("📚 Parameter Definitions"):
    st.markdown("""
    **Thermodynamic Parameters:**
    - **SBCAPE**: Surface-Based Convective Available Potential Energy
    - **MLCAPE**: Mixed Layer CAPE (50 hPa deep)
    - **MUCAPE**: Most Unstable CAPE
    - **CIN**: Convective Inhibition
    - **LCL**: Lifted Condensation Level height
    - **3CAPE**: CAPE within lowest 3 kilometers
    
    **Kinematic Parameters:**
    - **Wind Shear**: Bulk wind shear over specified depth
    - **SRH**: Storm-Relative Helicity (measures low-level rotation)
    - **Storm Motion**: Bunkers right-moving (RM) and left-moving (LM) vectors
    - **Effective Bulk Wind Difference**: Shear through effective inflow layer
    
    **Composite Parameters:**
    - **STPF**: Fixed-layer Supercell Tornado Parameter
    - **STPE**: Effective-layer Supercell Tornado Parameter  
    - Values >1 indicate increasing tornado potential
    - Values >3 indicate significant tornado potential
    - Values >8 indicate violent tornado potential
    """)

st.markdown("---")
st.markdown("*Comprehensive atmospheric analysis by Sekai Chandra (@Sekai_WX)*")