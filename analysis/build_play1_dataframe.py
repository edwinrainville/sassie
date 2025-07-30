import cftime 
from datetime import datetime, timedelta
from math import radians, cos, sin, asin, sqrt
import numpy as np
import netCDF4 as nc
import pandas as pd
import pyproj as proj
from scipy import io

from ice_edge_coordinate_transform import main as ice_edge_coordinate_transform
import sassie_tools

def get_wg_data(wg_data, variable):
    data = wg_data['SV3'][variable].squeeze()
    data = np.concatenate(data).flatten()
    return data

def datenum_to_datetime(datenum):
    """
    Convert Matlab datenum into Python datetime.
    :param datenum: Date in datenum format
    :return:        Datetime object corresponding to datenum.
    """
    days = datenum % 1
    return datetime.fromordinal(int(datenum)) \
           + timedelta(days=days) \
           - timedelta(days=366)

def compute_dist_to_ice_edge(lats, ice_edge_lat):
    dist_to_ice_edge = np.zeros(lats.size)
    
    for n in range(lats.size):
        # Basic calculation 
        diff = ice_edge_lat - lats[n]
        dist_to_ice_edge[n] = diff * 111 # 111 km per degree of lat

    return dist_to_ice_edge

def latlon_to_local(lat, lon, lat_0, lon_0):
    crs_wgs = proj.Proj(init='epsg:4326')  # assuming you're using WGS84 geographic

    #Erect own local flat cartesian coordinate system
    cust = proj.Proj("+proj=aeqd +lat_0={0} +lon_0={1} +datum=WGS84 +units=m".format(lat_0, lon_0))
    x, y = proj.transform(crs_wgs, cust, lon, lat)
    return x, y

def navigation_angle2math_angle(angle, convention='toward'):
    if convention == 'toward':
        math_angle = (450 - angle) % 360
    if convention == 'from':
        angle_toward = (angle + 180) % 360
        math_angle = (450 - angle_toward) % 360
    return math_angle

def main():
    """
    
    """

    # Load Play 1 SWIFT data
    # Create variables to store all data from different SWIFTs
    time = []
    latitude = []
    longitude = []
    hs_wave = []
    wave_direction = []
    wave_period = []
    drift_speed = []
    drift_direction = []
    instrument_type = []
    windspd = []
    winddir = []
    swiftnum = []
    salinity = []
    water_temperature = []
    swell_height = []
    swell_direction = []

    swift_fnames = ['./data/swift-data/SASSIE_Fall_2022_SWIFT12_play1.nc', './data/swift-data/SASSIE_Fall_2022_SWIFT13_play1.nc', 
                    './data/swift-data/SASSIE_Fall_2022_SWIFT15_play1.nc', './data/swift-data/SASSIE_Fall_2022_SWIFT16_play1.nc', 
                    './data/swift-data/SASSIE_Fall_2022_SWIFT17_play1.nc']

    for fname in swift_fnames:
        swift_data = nc.Dataset(fname)

        # Save the variables from each SWIFT file to new list
        time.append(cftime.num2pydate(swift_data['time'][:], units=swift_data['time'].units))
        latitude.append(swift_data['latitude'][:])
        longitude.append(swift_data['longitude'][:])
        hs_wave.append(swift_data['surface_wave_height'][:])
        wave_direction.append(swift_data['surface_wave_direction'][:])
        wave_period.append(swift_data['surface_wave_period'][:])
        drift_speed.append(swift_data['drift_speed'][:])
        drift_direction.append(swift_data['drift_direction'][:])
        instrument_type.append(['SWIFT' for n in range(swift_data['drift_direction'][:].size)])
        windspd.append(swift_data['wind_speed'][:])
        winddir.append(swift_data['wind_direction'][:])
        swiftnum.append(int(fname[-11:-9])*np.ones(swift_data['wind_direction'][:].size))
        salinity.append(swift_data['salinity'][:])
        water_temperature.append(swift_data['water_temperature'][:])

        # Compute Swell Height Based on obeserved frequency limits from plotted spectra (0.08 - 0.2 Hz)
        f_low = 0.08
        f_high = 0.2
        swell_height.append(sassie_tools.compute_sigwave_height_frequency_band(swift_data['wave_energy'][:], swift_data['frequency'][:], f_low, f_high))
        swell_direction_for_swift = np.empty(swift_data['time'][:].size)
        for n in range(swift_data['time'][:].size):
            swell_direction_for_swift[n] = sassie_tools.energy_weighted_direction(swift_data['wave_energy'][:,n], swift_data['frequency'][:], 
                                                                                  swift_data['spectral_directional_moment_east'][:,n], 
                                                                                  swift_data['spectral_directional_moment_north'][:,n], 
                                                                                  f_low, f_high) - 180
        swell_direction.append(swell_direction_for_swift)
        # Close the nc file
        swift_data.close()

    wave_glider_names = ['./data/L1/WaveGliders_L1/SV3-130_L1.mat', './data/L1/WaveGliders_L1/SV3-153_L1.mat', 
                        './data/L1/WaveGliders_L1/SV3-245_L1.mat', './data/L1/WaveGliders_L1/SV3-247_L1.mat']
    
    for fname in wave_glider_names:
        wg_data = io.loadmat(fname)
        # Save the variables from each SWIFT file to new list
        time_datenum = get_wg_data(wg_data, 'time')
        time_datetime = [datenum_to_datetime(time_datenum[n]) for n in range(time_datenum.size)]
        time.append(time_datetime)
        latitude.append(get_wg_data(wg_data, 'lat'))
        longitude.append(get_wg_data(wg_data, 'lon'))
        hs_wave.append(get_wg_data(wg_data, 'sigwaveheight'))
        wave_direction.append(np.nan*np.ones(get_wg_data(wg_data, 'lat').size))
        drift_speed.append(np.nan*np.ones(get_wg_data(wg_data, 'lat').size))
        drift_direction.append(np.nan*np.ones(get_wg_data(wg_data, 'lat').size))
        instrument_type.append(['WG' for n in range(get_wg_data(wg_data, 'lat').size)])
        windspd.append(get_wg_data(wg_data, 'windspd'))
        winddir.append(get_wg_data(wg_data, 'winddirT'))
        swiftnum.append(np.nan*np.ones(get_wg_data(wg_data, 'lat').size))
        wave_period.append(get_wg_data(wg_data, 'peakwaveperiod'))
        water_temperature.append(np.nan*np.ones(get_wg_data(wg_data, 'lat').size))
        salinity.append(np.nan*np.ones(get_wg_data(wg_data, 'lat').size))
        swell_height.append(np.nan*np.ones(get_wg_data(wg_data, 'lat').size))    
        swell_direction.append(np.nan*np.ones(get_wg_data(wg_data, 'lat').size))    
    
    # Concatenate and flatten the aggregate data
    time = np.concatenate(time).flatten()
    latitude = np.concatenate(latitude).flatten()
    longitude = np.concatenate(longitude).flatten()
    hs_wave = np.concatenate(hs_wave).flatten()
    wave_direction = np.concatenate(wave_direction).flatten()
    wave_period = np.concatenate(wave_period).flatten()
    drift_speed = np.concatenate(drift_speed).flatten()
    drift_direction = np.concatenate(drift_direction).flatten()
    instrument_type = np.concatenate(instrument_type).flatten()
    windspd = np.concatenate(windspd).flatten()
    winddir = np.concatenate(winddir).flatten()
    swiftnum = np.concatenate(swiftnum).flatten()
    salinity = np.concatenate(salinity).flatten()
    water_temperature = np.concatenate(water_temperature).flatten()
    swell_height = np.concatenate(swell_height).flatten()
    swell_direction = np.concatenate(swell_direction).flatten()

    # Compute the Local Cartesian Coordinate System
    lat_0 = 72.48
    lon_0 = -151.1
    x, y = latlon_to_local(latitude, longitude, lat_0, lon_0)

    # Create Pandas Dataframe to aggregate the SWIFT data
    df = pd.DataFrame(time, columns=['time'])
    df['latitude'] = latitude
    df['longitude'] = longitude
    df['hs'] = hs_wave
    df['wave_direction'] = wave_direction
    df['wave_direction_mathconv'] = 270 - wave_direction
    df['drift_speed'] = drift_speed
    df['drift_direction'] = drift_direction
    df['instrument_type'] = instrument_type
    df['windspd'] = windspd
    df['winddir'] = winddir
    df['swiftnum'] = swiftnum
    df['winddir_mathconv'] = 270 - winddir
    df['wave_period'] = wave_period
    df['salinity'] = salinity
    df['water_temperature'] = water_temperature
    df['swell_height'] = swell_height
    df['swell_direction'] = swell_direction
    df['x cartesian'] = x
    df['y cartesian'] = y

    # Fill all -999 with NaN values
    df.replace(to_replace=-999, value=np.nan, inplace=True)

    # Compute the x and y direction drift and wind speeds
    drift_direction_math_conv = navigation_angle2math_angle(df['drift_direction'], convention='toward')
    ew_drift_component = df['drift_speed'] * np.cos(np.deg2rad(drift_direction_math_conv))
    ns_drift_component = df['drift_speed'] * np.sin(np.deg2rad(drift_direction_math_conv))
    wind_direction_math_conv = navigation_angle2math_angle(df['winddir'], convention='from')
    ew_wind_speed = df['windspd'] * np.cos(np.deg2rad(wind_direction_math_conv))
    ns_wind_speed = df['windspd'] * np.sin(np.deg2rad(wind_direction_math_conv))
    df['EW_drift_speed'] = ew_drift_component
    df['NS_drift_speed'] = ns_drift_component
    df['EW_wind_speed'] = ew_wind_speed
    df['NS_wind_speed'] = ns_wind_speed

    # Last SWIFT deployed at 2022-09-09 19:30:00 and first SWIFT recovered at 2022-09-12 12:10:00
    mask = (df['time'] > '2022-09-09 19:30:00') & (df['time'] <= '2022-09-12 12:10:00')
    df = df.loc[mask]

    # Add seconds since start of the experiment
    start_time = pd.to_datetime('2022-09-09 19:30:00')
    df['seconds_since_start'] = (pd.to_datetime(df['time']) - start_time).dt.total_seconds()    

    # Save the dataframe
    df.to_csv('./data/play1_df.csv')

    # Run the coordinate transform script
    # ice_edge_coordinate_transform(lat_0, lon_0)

    return

if __name__ == "__main__":
    main()