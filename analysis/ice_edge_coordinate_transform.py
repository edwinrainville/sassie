from datetime import datetime, timedelta
import numpy as np
from scipy import io, interpolate
import pandas as pd
import pyproj as proj
from scipy import signal

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

def latlon_to_local(lat, lon, lat_0, lon_0):
    crs_wgs = proj.Proj(init='epsg:4326')  # assuming you're using WGS84 geographic

    #Erect own local flat cartesian coordinate system
    cust = proj.Proj("+proj=aeqd +lat_0={0} +lon_0={1} +datum=WGS84 +units=m".format(lat_0, lon_0))
    x, y = proj.transform(crs_wgs, cust, lon, lat)
    return x, y

def main(lat_0=72.48, lon_0=-151.1):
    """
    
    """
    # Load the Gridded Ice Maps
    ice_map_data = io.loadmat('../data/L1/nws_2022.mat')
    ice_map_lat = ice_map_data['LAT']
    ice_map_lon = ice_map_data['LON']
    ice_map_conc = ice_map_data['iceconc'] * 10
    ice_map_datenum = np.squeeze(ice_map_data['date'])
    ice_map_date = [datenum_to_datetime(ice_map_datenum[n].astype(np.float64)) for n in range(ice_map_datenum.size)]

    # Find index of closest Ice Map - September 10th 
    ind_for_ice_map = 252
    print(ice_map_date[ind_for_ice_map])
    ice_concentration = ice_map_conc[:,:,ind_for_ice_map]

    # Compute Ice Edge Based on 15% Concentration
    ice_conc_15percent = np.zeros(ice_concentration.shape)
    ice_conc_15percent[ice_concentration >= 15] = 1

    # Convert Lat lon on Ice Map to cartesian system 
    x_icemap, y_icemap = latlon_to_local(ice_map_lat, ice_map_lon, lat_0, lon_0)

    # Convert the SWIFT Track Coordinates to the local Cartesian system
    x_swifts_gliders, y_swifts_gliders = latlon_to_local(df['latitude'], df['longitude'], lat_0, lon_0)

    # Get time values for the SWIFTs
    time_swifts = df['time'][:]

    # Find the x and y location of the 15% ice concentration
    # Find the 15% concentration contour line
    ice_edge_contour_lon_vals = ice_map_lon[:,0]
    lat_vals = ice_map_lat[0,:]

    ice_edge_contour_lat_vals = []

    for n in range(ice_conc_15percent.shape[0]):
        ice_edge_lat_index_array = np.where(ice_conc_15percent[n,:] == 1)[0]
        if ice_edge_lat_index_array.size > 0:
            ice_edge_contour_lat_vals.append(lat_vals[ice_edge_lat_index_array[0]])
        else:
            ice_edge_contour_lat_vals.append(np.NaN)

    # Convert lon values to numpy array
    ice_edge_contour_lat_vals = np.array(ice_edge_contour_lat_vals)

    # Convert the ice edge values to cartesian and polar coordinates
    x_iceedge, y_iceedge = latlon_to_local(ice_edge_contour_lat_vals, ice_edge_contour_lon_vals, lat_0, lon_0)

    # Interpolate the x and y ice edge values to make a smooth curve of the ice edge
    x_iceedge_interp = np.linspace(-60000, 150000, num=50000)
    y_iceedge_interp = np.interp(x_iceedge_interp, x_iceedge, y_iceedge)

    fs = 1 / (x_iceedge_interp[1] - x_iceedge_interp[0])
    cutoff = 1/40000
    order = 1
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y_iceedge_smoothed = signal.filtfilt(b, a, y_iceedge_interp)

    #--------------------------------------------------------------------------------
    # Read in the sassie dataframe 
    df = pd.read_csv('./data/play1_df.csv')
    



    lat_lon_coords = create_pairs(df['latitude'], df['longitude'])

    # Transform the coordinates to the ice edge coordinate system
    along_ice_coords, cross_ice_coords = latlon_to_distances_along_across_line(lat_lon_coords, line_start, line_end)

    # Interpolate the ice concentration data at each measured point
    ice_conc_f = interpolate.RegularGridInterpolator((ice_map_lat[0,:], ice_map_lon[:,0]), np.transpose(ice_map_conc[:,:,ind_for_ice_map]), 
                                                  method='linear', bounds_error=False)
    # Ice Concentration at Each Point
    ice_conc_each_point = []
    for n in range(len(lat_lon_coords)):
        ice_conc_each_point.append(ice_conc_f(lat_lon_coords[n]).item())
    df['ice concentration'] = ice_conc_each_point

    # Save the coordinate transform to the dataframe
    df['along ice coord'] = along_ice_coords
    df['cross ice coord'] = cross_ice_coords
    df['ice edge bearing'] = np.ones(len(df['latitude'])) * ice_edge_bearing
    df['ice edge origin lon'] = np.ones(len(df['latitude'])) *  ice_edge_origin_lon 
    df['ice edge origin lat'] = np.ones(len(df['latitude'])) * ice_edge_origin_lat

    # Compute the Along and Cross ice drift speed
    df['along ice drift speed'] = df['drift_speed'] * np.cos(np.deg2rad(df['drift_direction'] - df['ice edge bearing']))
    df['cross ice drift speed'] = df['drift_speed'] * np.sin(np.deg2rad(df['drift_direction'] - df['ice edge bearing']))

    # Compute the Along and Cross ice drift speed
    df['along ice wind speed'] = df['windspd'] * np.cos(np.deg2rad((df['winddir']) - df['ice edge bearing']))
    df['cross ice wind speed'] = df['windspd'] * np.sin(np.deg2rad((df['winddir']) - df['ice edge bearing']))

    
 
    # Save the dataframe
    df.to_csv('./data/play1_df.csv')

    return

if __name__ == "__main__":
    main()