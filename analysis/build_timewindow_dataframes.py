import cartopy
import cv2
import cmocean
import cftime
from datetime import datetime, timedelta
import numpy as np
import netCDF4 as nc
import pandas as pd
import pyproj as proj
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.dates as mdates
import metpy.calc as mpcalc
from scipy import io, interpolate, stats, signal
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree


# Compute the Cross Ice and Along Ice Coordinates for each point
def compute_cross_ice_distance(point_x, point_y, ice_x, ice_y):
    """
    Computes the signed normal (perpendicular) distance of an array of points from the ice edge defined by two arrays of x and y coordinates.
    The distance is negative if the ice edge is above the point (ice_y > point_y).
    
    Parameters:
        point_x (array-like): An array of x coordinates for the points.
        point_y (array-like): An array of y coordinates for the points.
        ice_x (array-like): An array of x coordinates defining the ice edge.
        ice_y (array-like): An array of y coordinates defining the ice edge.
    
    Returns:
        numpy.ndarray: An array of signed distances corresponding to each point.
    """
    points = np.column_stack((point_x, point_y))
    ice_points = np.column_stack((ice_x, ice_y))
    
    # Use KDTree to find the closest point on the ice edge
    tree = cKDTree(ice_points)
    distances, indices = tree.query(points)
    
    # Determine the sign of the distance based on the y-coordinate comparison
    signed_distances = np.where(ice_y[indices] > point_y, -distances, distances)
    
    return signed_distances, indices

def latlon_to_local(lat, lon, lat_0, lon_0):
    crs_wgs = proj.Proj(init='epsg:4326')  # assuming you're using WGS84 geographic

    #Erect own local flat cartesian coordinate system
    cust = proj.Proj("+proj=aeqd +lat_0={0} +lon_0={1} +datum=WGS84 +units=m".format(lat_0, lon_0))
    x, y = proj.transform(crs_wgs, cust, lon, lat)
    return x, y

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

def smooth_ice_contour(lat_vals, lon_vals, lat_0, lon_0):
    # Convert the ice edge values to cartesian and polar coordinates
    x_iceedge, y_iceedge = latlon_to_local(lat_vals, lon_vals, lat_0, lon_0)

    # Interpolate the x and y ice edge values to make a smooth curve of the ice edge
    x_iceedge_interp = np.linspace(-100000, 200000, num=50000)
    y_iceedge_interp = np.interp(x_iceedge_interp, x_iceedge, y_iceedge)

    fs = 1 / (x_iceedge_interp[1] - x_iceedge_interp[0])
    cutoff = 1/40000
    order = 1
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y_iceedge_smoothed = signal.filtfilt(b, a, y_iceedge_interp)

    return x_iceedge_interp, y_iceedge_smoothed


def compute_window_wave_properties(df, x_ice_contour, y_ice_contour):
    # Average the Open ocean wave height
    H0 = np.nanmean(df[df['instrument_type']=='WG']['hs'])

    cross_ice_distance_initial, indices = compute_cross_ice_distance(df['x cartesian'], df['y cartesian'], x_ice_contour, y_ice_contour)

     
    # inds = ~np.isnan(df['hs'])
    # hs_nonans = [inds]
    # cross_ice_bin_center_nonans = cross_ice_bin_center[inds]

    # Define constants and initial guesses
    # out_of_ice_inds = cross_ice_bin_center_nonans < 0
    # H0 = np.nanmean(hs_nonans[out_of_ice_inds])

    # Fit the Decay Model
    initial_guess = [2.9*10**(-3), 10000]

    # Define the function to fit the data to
    def wave_height_decay_model(x, alpha, horizontal_offset, H0=H0):
        return H0 * np.exp(-alpha * (x - horizontal_offset))

    # Fit the exponential decay model for wave height to solve for decay rate and 
    x = cross_ice_distance_initial[cross_ice_distance_initial > 0]
    y = df['hs'][cross_ice_distance_initial > 0].values
    popt, _ = curve_fit(wave_height_decay_model, x, y, p0=initial_guess, nan_policy='omit') 
    alpha, horizontal_offset = popt

    # Compute the updated Cross Ice Distance 
    cross_ice_distance, indices = compute_cross_ice_distance(df['x cartesian'], df['y cartesian'], x_ice_contour, y_ice_contour + horizontal_offset) # Note that the offset is added to y even though it is x, the ice edge varies north and south, this is correct
    
    # Save All Constants to the dataframe
    df['cross ice distance'] = cross_ice_distance
    df['H0']= H0
    df['alpha'] = alpha
    df['horizontal offset'] = horizontal_offset
    df['hs_norm'] = df['hs']/H0
    df.loc[df['cross ice distance'] < 0]['hs_norm'] = 1

    # Save the ice edge


    return

def main():

    # load the play 1 dataframe 
    df = pd.read_csv('./data/play1_df.csv')

    # Get Ice Edge on each day # Load the Gridded Ice Maps
    ice_map_data = io.loadmat('./data/L1/nws_2022.mat')
    ice_map_lat = ice_map_data['LAT']
    ice_map_lon = ice_map_data['LON']
    ice_map_conc = ice_map_data['iceconc'] * 10
    ice_map_datenum = np.squeeze(ice_map_data['date'])
    ice_map_date = [datenum_to_datetime(ice_map_datenum[n].astype(np.float64)) for n in range(ice_map_datenum.size)]

    # Define a matrix to get the shape of ice concentration
    ice_concentration = ice_map_conc[:,:,0]

    # Day 1 - 1% concentration
    ice_conc_1percent_day1 = np.zeros(ice_concentration.shape)
    ice_conc_1percent_day1[ice_map_conc[:,:,252] >= 1] = 1

    # Day 2 - 1% Concentration
    ice_conc_1percent_day2 = np.zeros(ice_concentration.shape)
    ice_conc_1percent_day2[ice_map_conc[:,:,253]  >= 1] = 1

    # Day 3 - 1% Concentration
    ice_conc_1percent_day3 = np.zeros(ice_concentration.shape)
    ice_conc_1percent_day3[ice_map_conc[:,:,254] >= 1] = 1

    # Define the Local Cartesian Coordinate System
    lat_0 = 72.48
    lon_0 = -151

    # Get time values for the SWIFTs
    time_swifts = df['time'][:]
    datetimes = pd.to_datetime(time_swifts)
    date_numbers = np.squeeze(mdates.date2num(datetimes))

    # Find the x and y location of the 15% ice concentration
    # Find the 15% concentration contour line
    ice_edge_contour_lon_vals = ice_map_lon[:,0]
    lat_vals = ice_map_lat[0,:]

    ice_edge_1percent_day1_contour_lat_vals = []
    ice_edge_1percent_day2_contour_lat_vals = []
    ice_edge_1percent_day3_contour_lat_vals = []

    for n in range(ice_conc_1percent_day1.shape[0]):
        # day 1 
        ice_edge_lat_index_array = np.where(ice_conc_1percent_day1[n,:] == 1)[0]
        if ice_edge_lat_index_array.size > 0:
            ice_edge_1percent_day1_contour_lat_vals.append(lat_vals[ice_edge_lat_index_array[0]])
        else:
            ice_edge_1percent_day1_contour_lat_vals.append(np.NaN)
            
        # day 2
        ice_edge_lat_index_array = np.where(ice_conc_1percent_day2[n,:] == 1)[0]
        if ice_edge_lat_index_array.size > 0:
            ice_edge_1percent_day2_contour_lat_vals.append(lat_vals[ice_edge_lat_index_array[0]])
        else:
            ice_edge_1percent_day2_contour_lat_vals.append(np.NaN)

        # day 3
        ice_edge_lat_index_array = np.where(ice_conc_1percent_day3[n,:] == 1)[0]
        if ice_edge_lat_index_array.size > 0:
            ice_edge_1percent_day3_contour_lat_vals.append(lat_vals[ice_edge_lat_index_array[0]])
        else:
            ice_edge_1percent_day3_contour_lat_vals.append(np.NaN)

    # Convert lon values to numpy array
    ice_edge_1percent_day1_contour_lat_vals = np.array(ice_edge_1percent_day1_contour_lat_vals)
    ice_edge_1percent_day2_contour_lat_vals = np.array(ice_edge_1percent_day2_contour_lat_vals)
    ice_edge_1percent_day3_contour_lat_vals = np.array(ice_edge_1percent_day3_contour_lat_vals)

    # Smooth the ice concentration contours
    x_1percent_day1_contour, y_1percent_day1_contour = smooth_ice_contour(ice_edge_1percent_day1_contour_lat_vals, ice_edge_contour_lon_vals, lat_0, lon_0)
    x_1percent_day2_contour, y_1percent_day2_contour = smooth_ice_contour(ice_edge_1percent_day2_contour_lat_vals, ice_edge_contour_lon_vals, lat_0, lon_0)
    x_1percent_day3_contour, y_1percent_day3_contour = smooth_ice_contour(ice_edge_1percent_day3_contour_lat_vals, ice_edge_contour_lon_vals, lat_0, lon_0)


    # Create Time Windows
    # Window 1 df
    mask = (df['time'] <= '2022-09-10 00:00:00')
    window1_df = df.loc[mask]
    compute_window_wave_properties(window1_df, x_1percent_day1_contour, y_1percent_day1_contour)
    window1_df.to_csv('./data/play1_window1_df.csv')

    # window 2 df
    mask = (df['time'] > '2022-09-10 00:00:00') & (df['time'] <= '2022-09-10 06:00:00')
    window2_df = df.loc[mask]
    compute_window_wave_properties(window2_df, x_1percent_day1_contour, y_1percent_day1_contour)
    window2_df.to_csv('./data/play1_window2_df.csv')

    # window 3 df
    mask = (df['time'] > '2022-09-10 06:00:00') & (df['time'] <= '2022-09-10 12:00:00')
    window3_df = df.loc[mask]
    compute_window_wave_properties(window3_df, x_1percent_day1_contour, y_1percent_day1_contour)
    window3_df.to_csv('./data/play1_window3_df.csv')

    # window 4 df
    mask = (df['time'] > '2022-09-10 12:00:00') & (df['time'] <= '2022-09-10 18:00:00')
    window4_df = df.loc[mask]
    compute_window_wave_properties(window4_df, x_1percent_day1_contour, y_1percent_day1_contour)
    window4_df.to_csv('./data/play1_window4_df.csv')

    # window 5 df
    mask = (df['time'] > '2022-09-10 18:00:00') & (df['time'] <= '2022-09-11 00:00:00')
    window5_df = df.loc[mask]
    compute_window_wave_properties(window5_df, x_1percent_day2_contour, y_1percent_day2_contour)
    window5_df.to_csv('./data/play1_window5_df.csv')

    # window 6 df
    mask = (df['time'] > '2022-09-11 00:00:00') & (df['time'] <= '2022-09-11 06:00:00')
    window6_df = df.loc[mask]
    compute_window_wave_properties(window6_df, x_1percent_day2_contour, y_1percent_day2_contour)
    window6_df.to_csv('./data/play1_window6_df.csv')

    # window 7 df
    mask = (df['time'] > '2022-09-11 06:00:00') & (df['time'] <= '2022-09-11 12:00:00')
    window7_df = df.loc[mask]
    compute_window_wave_properties(window7_df, x_1percent_day2_contour, y_1percent_day2_contour)
    window7_df.to_csv('./data/play1_window7_df.csv')

    return

if __name__ == "__main__":
    main()