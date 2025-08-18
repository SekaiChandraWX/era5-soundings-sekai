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
            "year": [year],           # Use integer directly like original
            "month": [month],         # Use integer directly like original  
            "day": [day],             # Use integer directly like original
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
    """Process ERA5 netCDF file to extract sounding data"""
    progress_bar.progress(80, "Processing atmospheric data...")
    
    try:
        ds = Dataset(file_path)
        
        # Find nearest grid point
        latitudes = ds['latitude'][:]
        longitudes = ds['longitude'][:]
        x, y = np.abs(latitudes - lat).argmin(), np.abs(longitudes - lon).argmin()
        
        # Extract variables
        levels = ds['pressure_level'][:]
        z = ds['z'][0, :, y, x] / 9.81  # Convert geopotential to height
        u = ds['u'][0, :, y, x]
        v = ds['v'][0, :, y, x]
        temp = ds['t'][0, :, y, x] - 273.15  # Convert to Celsius
        rh = ds['r'][0, :, y, x]
        
        # Calculate derived variables
        wdir = np.degrees(np.arctan2(u, v)) + 180
        wspd = np.hypot(u, v) * 1.944  # Convert to knots
        
        # Calculate dewpoint
        dew = np.asarray(
            mpcalc.dewpoint_from_relative_humidity(
                np.asarray(temp) * units.degC, 
                np.asarray(rh) * units.percent
            )
        )
        
        # Get valid time
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
    """Create simple sounding plot and save as JPG"""
    progress_bar.progress(85, "Creating sounding plot...")
    
    try:
        # Create temporary file path for the plot
        plot_filename = f"era5_sounding_{int(time.time())}.jpg"
        temp_plot_path = os.path.join(tempfile.gettempdir(), plot_filename)
        
        # Use your original plotting approach
        create_simple_sounding_plot(data, temp_plot_path, lat)
        
        # Read the image file and return as matplotlib figure for Streamlit
        from PIL import Image
        img = Image.open(temp_plot_path)
        
        # Convert PIL image to matplotlib figure
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.imshow(img)
        ax.axis('off')
        
        # Clean up temp file
        if os.path.exists(temp_plot_path):
            os.remove(temp_plot_path)
            
        return fig
        
    except Exception as e:
        raise Exception(f"Plot creation failed: {str(e)}")

def create_simple_sounding_plot(data, file_path, lat):
    """Create sounding plot using original working logic"""
    try:
        # Convert ERA5 data to arrays (like original)
        pressure = np.array(data['levels'])
        height = np.array(data['z'])
        temperature = np.array(data['temp'])
        dewpoint = np.array(data['dew'])
        u_wind = np.array(data['u']) * 1.944  # Convert to knots
        v_wind = np.array(data['v']) * 1.944
        
        # Sort by decreasing pressure (surface to top)
        sort_idx = np.argsort(pressure)[::-1]
        pressure = pressure[sort_idx]
        height = height[sort_idx]
        temperature = temperature[sort_idx] 
        dewpoint = dewpoint[sort_idx]
        u_wind = u_wind[sort_idx]
        v_wind = v_wind[sort_idx]
        
        # Filter valid data
        valid_mask = (
            (pressure > 0) &
            (np.isfinite(pressure)) & (np.isfinite(temperature)) & (np.isfinite(dewpoint)) &
            (temperature > -100) & (temperature < 60) &
            (dewpoint > -100) & (dewpoint < 60) &
            (dewpoint <= temperature + 0.1)
        )
        
        # Apply mask and add units
        p = pressure[valid_mask] * units.hPa
        T = temperature[valid_mask] * units.degC
        Td = dewpoint[valid_mask] * units.degC
        u = u_wind[valid_mask] * units.knot
        v = v_wind[valid_mask] * units.knot
        
        # Calculate basic parameters
        try:
            # Surface-based parcel
            sb_parcel_prof = mpcalc.parcel_profile(p, T[0], Td[0])
            sb_cape, sb_cin = mpcalc.cape_cin(p, T, Td, sb_parcel_prof)
            
            # LCL
            lcl_p, lcl_t = mpcalc.lcl(p[0], T[0], Td[0])
        except:
            sb_cape = sb_cin = 0 * units('J/kg')
            lcl_p, lcl_t = p[0], T[0]
        
        # Determine hemisphere
        is_southern_hemisphere = lat < 0
        hemisphere_text = "SHEM" if is_southern_hemisphere else "NHEM"
        
        # === SIMPLE PLOTTING ===
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(16, 10), facecolor='#2F2F2F')
        
        # Create Skew-T plot
        skew = SkewT(fig, rotation=45)
        skew.ax.set_facecolor('#2F2F2F')
        
        # Plot temperature and dewpoint profiles
        skew.plot(p, T, 'r', linewidth=2.5, label='Temperature')
        skew.plot(p, Td, 'g', linewidth=2.5, label='Dewpoint')
        
        # Plot wind barbs
        skew.plot_barbs(p[::3], u[::3], v[::3], barbcolor='white', flagcolor='white')
        
        # Plot LCL
        skew.plot(lcl_p, lcl_t, 'wo', markerfacecolor='white', markeredgecolor='black', markersize=8)
        
        # Plot parcel profile
        try:
            prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')
            skew.plot(p, prof, color='#FFFF99', linewidth=2, linestyle='--')
            skew.shade_cin(p, T, prof, Td, alpha=0.4)
            skew.shade_cape(p, T, prof, alpha=0.4)
        except:
            pass
        
        # Styling
        skew.ax.set_ylim(1000, 100)
        skew.ax.set_xlim(-40, 60)
        skew.ax.set_xlabel('Temperature (°C)', color='white')
        skew.ax.set_ylabel('Pressure (hPa)', color='white')
        skew.ax.tick_params(colors='white')
        skew.ax.grid(True, alpha=0.3, color='white')
        
        # Add reference lines
        skew.ax.axvline(0, color='white', linestyle='--', linewidth=2, alpha=0.7)
        skew.plot_dry_adiabats(colors='white', alpha=0.3)
        skew.plot_moist_adiabats(colors='white', alpha=0.3)
        skew.plot_mixing_lines(colors='white', alpha=0.3)
        
        # Title
        formatted_date = data['valid_date'].strftime("%Y-%m-%d %H:%M UTC")
        skew.ax.set_title(f'ERA5 Atmospheric Sounding ({hemisphere_text}) - {formatted_date}',
                         fontsize=14, fontweight='bold', color='white', pad=20)
        
        # Add basic parameters as text
        param_text = f"SBCAPE: {sb_cape:~.0f}    SBCIN: {sb_cin:~.0f}"
        plt.figtext(0.5, 0.08, param_text, ha='center', fontsize=12, color='white')
        
        # Attribution
        plt.figtext(0.5, 0.02, 'Plotted by Sekai Chandra (@Sekai_WX)',
                   ha='center', fontsize=8, style='italic', color='white')
        
        # Save as JPG
        plt.tight_layout()
        plt.savefig(file_path, format='jpg', bbox_inches='tight', 
                   facecolor='#2F2F2F', dpi=150)
        plt.close()
        
    except Exception as e:
        raise Exception(f"Simple plot creation failed: {str(e)}")

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
            
            # Create plot
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

# Streamlit UI
st.title("🌪️ ERA5 Atmospheric Sounding Plotter")
st.markdown("### Comprehensive Atmospheric Profile Analysis")

if not DEPENDENCIES_AVAILABLE:
    st.error("❌ Required atmospheric science libraries are not installed.")
    st.info("This tool requires: netCDF4, metpy, and cdsapi")
    st.stop()

st.markdown("""
Generate professional **Skew-T log-P diagrams** with **hodographs** and **severe weather parameters** 
using **ERA5 reanalysis data**. Includes hemisphere-specific storm motion calculations and 
comprehensive hazard assessment. No registration required - just select your location and date!
""")

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
        
        os.environ['CDSAPI_URL'] = 'https://cds.climate.copernicus.eu/api'  # Correct API URL
        os.environ['CDSAPI_KEY'] = cds_key
        return True
        
    except Exception as e:
        st.error(f"❌ CDS API setup failed: {e}")
        return False

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
    
    generate_button = st.button("🚀 Generate Sounding Analysis", type="primary")

with col2:
    st.subheader("📊 Atmospheric Sounding")
    
    if generate_button:
        if lat is not None and lon is not None:
            try:
                fig, data = process_era5_sounding(date_input, hour_input, lat, lon)
                
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                gc.collect()
                
                st.success("✅ ERA5 sounding analysis complete!")
                st.info("💡 Right-click on the plot to save it to your device.")
                
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
    
    **Sounding Analysis Features:**
    - **Skew-T log-P diagrams** with temperature, dewpoint, and wind profiles
    - **Hodographs** showing wind patterns by altitude with storm motion vectors
    - **CAPE/CIN calculations** for convective potential assessment
    - **Wind shear analysis** at multiple levels (0-1km, 0-3km, 0-6km, 0-8km)
    - **Storm-Relative Helicity (SRH)** with hemisphere-specific calculations
    - **Supercell Tornado Parameter (STP)** for severe weather potential
    - **Automated hazard assessment** (TOR, SVR, MRGL, etc.)
    
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
    
    **Kinematic Parameters:**
    - **Wind Shear**: Bulk wind shear over specified depth
    - **SRH**: Storm-Relative Helicity (measures low-level rotation)
    - **Storm Motion**: Bunkers right-moving (RM) and left-moving (LM) vectors
    
    **Composite Parameters:**
    - **STP**: Supercell Tornado Parameter (combines CAPE, shear, SRH, LCL)
    - Values >1 indicate increasing tornado potential
    - Values >3 indicate significant tornado potential
    """)

with st.expander("🔧 Technical Details"):
    st.markdown("""
    **CDS API Configuration:**
    - Uses Copernicus Climate Data Store (CDS) API
    - API endpoint: `https://cds.climate.copernicus.eu/api`
    - Requires valid CDS account and API key
    
    **Data Processing:**
    - Downloads ERA5 pressure level data in NetCDF format
    - Extracts nearest grid point to requested location
    - Calculates comprehensive atmospheric parameters
    - Generates professional-quality visualizations
    
    **Performance:**
    - Small geographic area (0.02° x 0.02°) for fast downloads
    - All 37 pressure levels (1000-1 hPa) included
    - Processing optimized for Streamlit cloud environment
    """)

with st.expander("⚠️ Data Availability"):
    st.markdown("""
    **Temporal Coverage:**
    - **Historical**: January 1940 - Present
    - **Real-time delay**: ~5 days behind current date
    - **Analysis times**: Hourly (00-23 UTC)
    
    **Spatial Resolution:**
    - **Horizontal**: ~31 km (0.28° x 0.28°)
    - **Vertical**: 37 pressure levels (1000-1 hPa)
    
    **Processing Notes:**
    - ERA5 data is automatically downloaded from Copernicus CDS
    - Processing time: 1-3 minutes for hourly data
    - ERA5 is a **reanalysis product**, not real-time observations
    - Small-scale features may be smoothed compared to radiosonde data
    - Best used for **climatological analysis** and **case studies**
    """)

st.markdown("---")
st.markdown("*Comprehensive atmospheric analysis by Sekai Chandra (@Sekai_WX)*")