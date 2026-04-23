# -*- coding: utf-8 -*-
import os
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from .setup import allowed_moments

from .data_keys import inuse_dtype
from .data_keys import dat_key_prefix, inuse_key_prefix, std_key_prefix, err_key_prefix

from . import variance_averaging

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from .sps import interpolate_df2_on_df1, calcUTMcoordinates, calcEpochTime, read_concat_DGPS_sps_files
import copy
import libaarhusxyz
import time


def build_inuse_dataframe(data, channel):
    """
    Function to build a fresh, fully in-use dataframe
    """
    str_channel = f"0{channel}"[-2:]
    data_key = f"{dat_key_prefix}{str_channel}"
    inuse_key = f"{inuse_key_prefix}{str_channel}"

    try:
        inuse_df = pd.DataFrame(np.ones(data.layer_data[inuse_key].shape),
                                dtype=data.layer_data[inuse_key].dtypes[0],
                                index=data.layer_data[inuse_key].index,
                                columns=data.layer_data[inuse_key].columns)
    except:
        inuse_df = pd.DataFrame(np.ones(data.layer_data[data_key].shape),
                                dtype=inuse_dtype,
                                index=data.layer_data[data_key].index,
                                columns=data.layer_data[data_key].columns)
    return inuse_df


def is_dual_moment(gex, verbose=False):
    if 'Channel2' in gex.gex_dict.keys():
        if verbose: print('looks like dual moment data.')
        dual_moment=True
    else:
        if verbose: print('looks like single moment data.')
        dual_moment=False
    return dual_moment


def get_flightline_bbox(Data, pos_keys=['UTMX','UTMY']):
    df_points = Data.flightlines
    return (df_points[pos_keys[0]].min(),
            df_points[pos_keys[1]].min(), 
            df_points[pos_keys[0]].max(), 
            df_points[pos_keys[1]].max())

def read_shape_in_margins(shapefile, bbox, crs, margin_x=2000, margin_y=2000):
    bbox=(bbox[0]-margin_x,
          bbox[1]-margin_y, 
          bbox[2]+margin_x, 
          bbox[3]+margin_y)
          
    # read shp into geopandas geodataframe:
    df_shp=gpd.read_file(shapefile, bbox=bbox)
    return df_shp.to_crs(crs), bbox

def getGateTimesFromGEX(gex, channel='Channel1'):
    NoGates=int(gex.gex_dict[channel]['NoGates'])
    if 'RemoveGatesFrom' in gex.gex_dict[channel].keys():
        RemoveGatesFrom=int(gex.gex_dict[channel]['RemoveGatesFrom'])
    else:
        RemoveGatesFrom=int(0)
    if not('MeaTimeDelay' in gex.gex_dict[channel].keys()):
        gex.gex_dict[channel]['MeaTimeDelay']=0.0
    gatetimes=(gex.gex_dict['General']['GateTime'][RemoveGatesFrom:NoGates,:] + gex.gex_dict[channel]['GateTimeShift'] + gex.gex_dict[channel]['MeaTimeDelay'] )
    return gatetimes

def inuse_moment(moment):
    return 'InUse_'+moment.split('_')[-1]

def errKey(moment):
    return 'relErr_'+moment.split('_')[-1]

def stdKey(moment):
    return 'STD_'+moment.split('_')[-1]


def resampleWaveform(gex):
    t=gex.General['WaveformLMPoint'][:,0]
    a=gex.General['WaveformLMPoint'][:,1]
    ti=np.linspace(t.min(), t.max(), 100000)
    f=interp1d(t, a, kind='linear')
    ai=f(ti)
    gex.General['WaveformLMPointInterpolated']=np.vstack([ti, ai]).T
    dti=  ti[:-1]+np.diff(ti)/2
    dai=np.diff(ai)
    gex.General['WaveformLMPointInterpolatedGrad']=np.vstack([dti, dai]).T
    d2ti=dti[:-1]+np.diff(dti)/2
    d2ai=np.diff(dai)
    gex.General['WaveformLMPointInterpolatedGradGrad']=np.vstack([d2ti, d2ai]).T
    return gex

def estimateInlineSamplig(data):
    if 'line' in data.flightlines.keys():
        line_key='line'
    elif 'line_no' in data.flightlines.keys():
        line_key='line_no'
    elif 'Line' in data.flightlines.keys():
        line_key='Line'
    lines = splitData_lines(data, line_key=line_key)
    med_dist=[]
    for key in lines.keys():
        if not('lineoffset' in lines[key].flightlines.columns):
            calc_lineOffset(lines[key])
        med_dist.append(lines[key].flightlines.lineoffset.diff().median())
    return np.median(np.array(med_dist))

def get_line(Data, line_no, line_key='line'):
    model_dict = {}
    key_list=list(Data.model_dict.keys())
    key_list.remove('layer_data')
    key_list.remove('flightlines')
    for key in key_list:
        model_dict[key]=Data.model_dict[key]
    # line spceific data:
    idx = Data.flightlines[line_key]==line_no
    model_dict["flightlines"] = Data.flightlines[idx]
    model_dict["layer_data"] = {}
    for key in Data.layer_data.keys():
        model_dict["layer_data"][key]=Data.layer_data[key][idx]
    return libaarhusxyz.XYZ(model_dict, normalize=False)

def split_lines(data, synth, model, line_key='line'):
    line_nos = np.unique(data.flightlines[line_key])
    lines={}
    for line_no in line_nos:
        line={"data" : get_line(data, line_no, line_key=line_key),
              "synth": get_line(synth, line_no, line_key=line_key),
              "model": get_line(model, line_no, line_key=line_key)}
        lines[str(line_no)] = line
    return lines


def splitData_lines(Data, line_key='line'):
    line_nos=Data.flightlines[line_key].unique()
    #line_nos = np.unique(Data.flightlines[line_key].values)
    Lines={}
    for line_no in line_nos:
        Lines[line_no] = get_line(Data, line_no, line_key=line_key)
    return Lines


def splitModel_lines(model, line_key='line'):
    return splitData_lines(model, line_key=line_key)


def merge_lines(lines_dict):
    return libaarhusxyz.XYZ(*lines_dict.values())

def concatXYZ(xyz1, xyz2):
    return libaarhusxyz.XYZ(xyz1, xyz2)

def calc_lineOffset(data):
    if ('UTMX' in data.flightlines.keys()) and ('UTMY' in data.flightlines.keys()):
        pos_keys=['UTMX', 'UTMY']
    elif ('utmx' in data.flightlines.keys()) and ('utmy' in data.flightlines.keys()):
        pos_keys=['utmx', 'utmy']
    else:
        raise Exception("Sorry, no coordinates with labels UTMX, UTMY or utmx, utmy, found in data") 
    for key in ['line', 'line_no', 'Line']:
        if key in data.flightlines.keys():
            line_key=key
    data.flightlines.insert(len(data.flightlines.columns), 'lineoffset', np.zeros( len(data.flightlines) ) )    
    for line in data.flightlines[line_key].unique():
        filt=data.flightlines[line_key]==line
        
        dx=data.flightlines.loc[filt, pos_keys[0]].diff()
        dx.iloc[0]=0
        dy=data.flightlines.loc[filt, pos_keys[1]].diff()
        dy.iloc[0]=0
        data.flightlines.loc[filt, 'lineoffset']=np.cumsum( np.sqrt(dx**2 + dy**2) )


def scaleData(processing, scalefactors=[1, 1]):
    data = processing.xyz
    for n, moment in enumerate(allowed_moments):
        if moment in data.layer_data.keys():
            scaled_key='Gate_scaled_'+moment.split('Gate_')[-1]
            data.layer_data[scaled_key]=data.layer_data[moment]*scalefactors[n]
        # if hasattr(processing, 'Noise'):
        #     if moment in processing.Noise.keys():
        #         processing.Noise[moment]=processing.Noise[moment]*scalefactors[n]
                

def unscaleData(data):
    for moment in allowed_moments:
        if moment in data.layer_data.keys():
            # scaling
            channel=moment.split('_')[-1]
            scaled_key='Gate_scaled_'+channel
            dipole_moment_key='DipoleMoment_'+channel
            dB_dt_df=data.layer_data[moment]
            M_df=pd.DataFrame(data=np.tile(data.flightlines[dipole_moment_key],[dB_dt_df.shape[1], 1]).T,
                              index=dB_dt_df.index,
                              columns=dB_dt_df.columns)
            data.layer_data[scaled_key]=dB_dt_df * data.model_info['scalefactor'] * M_df

def dBdt_to_rhoa(processing):
    data = processing.xyz
    mu0= 4 * np.pi * 1e-7
    for moment in allowed_moments:
        if moment in data.layer_data.keys():
            rhoa_key='Rhoa_' + moment.split('Gate_')[-1]
            dB_dt_df=data.layer_data[moment]
            gate_times_df=pd.DataFrame(data=np.tile(processing.GateTimes[moment], [dB_dt_df.shape[0],1]),
                                    index=dB_dt_df.index,
                                    columns=dB_dt_df.columns)
            M_df=pd.DataFrame(data=np.ones(data.layer_data[moment].shape) / data.model_info['scalefactor'],
                              index=dB_dt_df.index,
                              columns=dB_dt_df.columns)
            data.layer_data[rhoa_key] = 1/np.pi * (M_df / (20 * dB_dt_df)  )**(2/3) *   (mu0 / gate_times_df)**(5/3)


def sampleDEM(DEMfilename,
              df,
              poskeys=['utmx', 'utmy'],
              z_key='zdem_laser1',
              crs=None):
    dem = rasterio.open(DEMfilename, 'r')
    dem_xmin, dem_ymin, dem_xmax, dem_ymax = dem.bounds

    if crs is None:
        print('Assuming that the DTM projection is the same as the AEM data')
        crs = dem.crs

    if type(crs) is int:
        dst_crs = f"EPSG:{crs}"
    elif 'epsg:' in str(crs):
        dst_crs = crs.replace('epsg', 'EPSG')
    elif 'EPSG:' in str(crs):
        dst_crs = crs

    assert 'dst_crs' in locals(), f'There is something wrong with the supplied crs parameter ({crs}).'
    dst_crs = rasterio.crs.CRS.from_string(dst_crs)

    xmin = df[poskeys[0]].min()
    xmax = df[poskeys[0]].max()
    ymin = df[poskeys[1]].min()
    ymax = df[poskeys[1]].max()

    geometry = [Point(xy) for xy in zip(df[poskeys[0]], df[poskeys[1]])]
    geo_df = gpd.GeoDataFrame(df, crs=dst_crs, geometry=geometry)

    if dem.crs != crs:
        print(f"  - DTM projection ({dem.crs}) is not the same as specified projection ({dst_crs}).")
        print(f"      Re-projecting the data to {dem.crs} for sampling only.")
        geo_df.to_crs(epsg=dem.crs.to_epsg(), inplace=True)

        new_poskeys = copy.copy(poskeys)
        if ':' in str(dem.crs):
            new_poskeys[0] = f"{new_poskeys[0]}_{str(dem.crs).split(':')[1]}"
            new_poskeys[1] = f"{new_poskeys[1]}_{str(dem.crs).split(':')[1]}"
        else:
            new_poskeys[0] = f"{new_poskeys[0]}_{str(dem.crs)}"
            new_poskeys[1] = f"{new_poskeys[1]}_{str(dem.crs)}"

        df[new_poskeys[0]] = geo_df.geometry.x
        df[new_poskeys[1]] = geo_df.geometry.y

        xmin = df[new_poskeys[0]].min()
        xmax = df[new_poskeys[0]].max()
        ymin = df[new_poskeys[1]].min()
        ymax = df[new_poskeys[1]].max()

    for a, b in zip([    xmin,     ymin, dem_xmax, dem_ymax],
                    [dem_xmin, dem_ymin,     xmax,     ymax]):
        if a < b:
            bounds_comparison = pd.DataFrame([[dem_xmin, dem_ymin, dem_xmax, dem_ymax],
                                              [    xmin,     ymin,     xmax,     ymax]],
                                             index=['orig_dem', 'reprojected_dem', 'flightlines'],
                                             columns=['left', 'top', 'right', 'bottom'])
            print(bounds_comparison)
            raise Exception(f'coordinates outside raster bounds for projection: {dem.crs}')
        # else:
            # print(f'coordinates are inside raster bounds for projection: {dem.crs}')

    coord_list = [(x, y) for x, y in zip(geo_df["geometry"].x, geo_df["geometry"].y)]
    dtm_values = [x[0] for x in dem.sample(coord_list)]
    dem.close()

    df[z_key] = dtm_values


def build_l10_dBdt_time_df(processing, data_key):
    scalefactor = processing.xyz.model_info['scalefactor']
    data_df = copy.copy(processing.xyz.layer_data[data_key])
    gate_times = copy.copy(processing.GateTimes[data_key])

    dBdt_df = np.abs(data_df) * scalefactor
    l10_dBdt_df = np.log10(dBdt_df)

    # FIXME: This may need fixed better in the future
    gate_times[gate_times < 0] = np.nan
    l10_gate_times = np.log10(gate_times)
    l10_gate_times_df = pd.DataFrame(np.tile(l10_gate_times, (processing.xyz.layer_data[data_key].shape[0], 1)))

    return l10_dBdt_df, l10_gate_times_df


def _needs_regression(l10_times, threshold=0.02):
    """Return True if any adjacent gate spacing in log10 is below threshold.

    Used by method='auto' to decide between adjacent finite difference
    and regression-based slope/curvature computation.
    """
    valid = l10_times[np.isfinite(l10_times)]
    if len(valid) < 2:
        return False
    spacings = np.diff(valid)
    return bool(np.any(spacings < threshold))


def _build_adaptive_windows(l10_times, min_gates=5, max_half_decades=0.15):
    """Build per-gate windows for regression: at least min_gates, up to max_half_decades.

    For tightly-spaced early gates, the decade-based window captures many gates.
    For widely-spaced late gates, we guarantee at least min_gates.

    Parameters
    ----------
    l10_times : array-like
        1D array of log10(gate_times).
    min_gates : int
        Minimum number of gates in each window.
    max_half_decades : float
        Maximum half-width of the window in decades of log10(t).

    Returns
    -------
    list of ndarray
        Per-gate arrays of indices into l10_times.
    """
    n = len(l10_times)
    windows = []
    for k in range(n):
        # Start with decade-based window
        decade_mask = np.abs(l10_times - l10_times[k]) <= max_half_decades
        indices = np.where(decade_mask)[0]

        # Ensure at least min_gates (expand symmetrically from k)
        if len(indices) < min_gates:
            left = k
            right = k
            while (right - left + 1) < min_gates:
                if left > 0:
                    left -= 1
                if (right - left + 1) >= min_gates:
                    break
                if right < n - 1:
                    right += 1
            indices = np.arange(left, right + 1)

        windows.append(indices)
    return windows


def _adjacent_slopes(l10_dBdt_df, l10_times_1d, original_data):
    """Compute slopes using adjacent-gate finite difference (original method).

    Equivalent to the old .diff() formula with sign-change guard.
    """
    slope = l10_dBdt_df.diff(axis=1) / pd.DataFrame(
        np.tile(np.diff(l10_times_1d, prepend=np.nan), (len(l10_dBdt_df), 1)),
        index=l10_dBdt_df.index, columns=l10_dBdt_df.columns
    )

    # Sign change guard
    prev_data = original_data.shift(1, axis=1)
    bad = (
        (original_data * prev_data < 0)
        | (original_data == 0)
        | (prev_data == 0)
    )
    slope[bad] = np.nan

    return slope


def _regression_slopes(l10_dBdt_df, l10_times_1d, min_gates=5,
                       max_half_decades=0.15, min_points=3):
    """Compute slopes using OLS linear regression over adaptive windows.

    For each gate, all soundings with fully valid windows are solved in a
    single vectorized ``np.linalg.lstsq`` call.  Soundings with partial
    NaN are handled individually.
    """
    l10_dBdt_arr = l10_dBdt_df.values
    n_soundings, n_gates = l10_dBdt_arr.shape
    slope_arr = np.full((n_soundings, n_gates), np.nan)

    windows = _build_adaptive_windows(l10_times_1d, min_gates, max_half_decades)

    for k in range(n_gates):
        indices = windows[k]
        t_local = l10_times_1d[indices]
        y_all = l10_dBdt_arr[:, indices]  # (n_soundings, n_window)

        # Design matrix for linear fit: [t, 1]
        A = np.column_stack([t_local, np.ones(len(t_local))])

        # Soundings where every gate in the window is valid
        all_valid = np.all(np.isfinite(y_all), axis=1)

        if all_valid.any():
            Y = y_all[all_valid].T  # (n_window, n_batch)
            coeffs, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
            slope_arr[all_valid, k] = coeffs[0]  # slope coefficients

        # Handle partial-NaN soundings individually
        partial = ~all_valid & (np.isfinite(y_all).sum(axis=1) >= min_points)
        for i in np.where(partial)[0]:
            valid = np.isfinite(y_all[i])
            A_sub = np.column_stack([t_local[valid], np.ones(valid.sum())])
            coeffs_i, _, _, _ = np.linalg.lstsq(A_sub, y_all[i, valid],
                                                 rcond=None)
            slope_arr[i, k] = coeffs_i[0]

    return pd.DataFrame(slope_arr, index=l10_dBdt_df.index,
                        columns=l10_dBdt_df.columns)


def _adjacent_curvatures(l10_dBdt_df, l10_times_1d, original_data):
    """Compute curvatures using adjacent-gate finite difference (original method).

    Uses the 3-point formula: (f[k+1] - 2*f[k] + f[k-1]) / (t[k+1] - t[k-1])^2
    with sign-change guard masking gates where any of the 3 endpoints is <= 0.
    """
    n_gates = len(l10_times_1d)
    curvature = pd.DataFrame(np.nan, index=l10_dBdt_df.index,
                             columns=l10_dBdt_df.columns)

    for k in range(1, n_gates - 1):
        if np.isnan(l10_times_1d[k - 1]) or np.isnan(l10_times_1d[k + 1]):
            continue
        t_span = l10_times_1d[k + 1] - l10_times_1d[k - 1]
        curvature.iloc[:, k] = (
            l10_dBdt_df.iloc[:, k + 1] - 2 * l10_dBdt_df.iloc[:, k]
            + l10_dBdt_df.iloc[:, k - 1]
        ) / (t_span ** 2)

    # Sign change guard
    for k in range(1, n_gates - 1):
        if np.isnan(l10_times_1d[k - 1]) or np.isnan(l10_times_1d[k + 1]):
            continue
        bad = (
            (original_data.iloc[:, k - 1] <= 0)
            | (original_data.iloc[:, k] <= 0)
            | (original_data.iloc[:, k + 1] <= 0)
        )
        curvature.loc[bad, curvature.columns[k]] = np.nan

    return curvature


def _regression_curvatures(l10_dBdt_df, l10_times_1d, min_gates=7,
                           max_half_decades=0.15, min_points=5):
    """Compute curvatures via OLS quadratic fit over adaptive windows.

    The raw second derivative from the quadratic fit (2*a) is normalized to
    match the adjacent-gate curvature convention used by SkyTEM processing.
    The adjacent formula computes ``(f[k+1] - 2*f[k] + f[k-1]) / span**2``
    where ``span = t[k+1] - t[k-1]``.  For a quadratic with coefficient *a*
    evaluated at equally-spaced points with half-span *h = span/2*, this
    equals ``a/2``.  To make the regression output comparable, we scale
    the true second derivative by ``(half_span)**2``::

        normalized_curvature = 2 * a * (window_span / 2) ** 2

    This ensures that the same curvature threshold (e.g. 10) is meaningful
    regardless of whether the data has tightly-spaced gates (HeliTEM) or
    widely-spaced gates (SkyTEM).

    For each gate, all soundings with fully valid windows are solved in a
    single vectorized ``np.linalg.lstsq`` call.  Soundings with partial
    NaN are handled individually.
    """
    l10_dBdt_arr = l10_dBdt_df.values
    n_soundings, n_gates = l10_dBdt_arr.shape
    curv_arr = np.full((n_soundings, n_gates), np.nan)

    windows = _build_adaptive_windows(l10_times_1d, min_gates, max_half_decades)

    for k in range(n_gates):
        indices = windows[k]
        t_local = l10_times_1d[indices]
        y_all = l10_dBdt_arr[:, indices]  # (n_soundings, n_window)
        window_half_span = (t_local[-1] - t_local[0]) / 2.0

        # Design matrix for quadratic fit: [t^2, t, 1]
        A = np.column_stack([t_local ** 2, t_local, np.ones(len(t_local))])

        # Soundings where every gate in the window is valid
        all_valid = np.all(np.isfinite(y_all), axis=1)

        if all_valid.any():
            Y = y_all[all_valid].T  # (n_window, n_batch)
            coeffs, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
            curv_arr[all_valid, k] = 2 * coeffs[0] * window_half_span ** 2

        # Handle partial-NaN soundings individually
        partial = ~all_valid & (np.isfinite(y_all).sum(axis=1) >= min_points)
        for i in np.where(partial)[0]:
            valid = np.isfinite(y_all[i])
            t_v = t_local[valid]
            A_sub = np.column_stack([t_v ** 2, t_v, np.ones(valid.sum())])
            coeffs_i, _, _, _ = np.linalg.lstsq(A_sub, y_all[i, valid],
                                                 rcond=None)
            curv_arr[i, k] = 2 * coeffs_i[0] * window_half_span ** 2

    return pd.DataFrame(curv_arr, index=l10_dBdt_df.index,
                        columns=l10_dBdt_df.columns)


def calculate_transient_slopes(processing, data_key, method='auto'):
    """Compute transient decay slopes in log10(|dBdt|) vs log10(t) space.

    Parameters
    ----------
    processing : object
        Processing object with xyz data and GateTimes.
    data_key : str
        Key for the gate data in layer_data.
    method : str, default 'auto'
        'adjacent'   - finite difference between adjacent gates (original method)
        'regression' - OLS regression over adaptive windows
        'auto'       - use 'adjacent' if gate spacing is wide, 'regression' if tight

    Returns
    -------
    pd.DataFrame
        Slope values with same index/columns as the input data.
    """
    l10_dBdt_df, l10_gate_times_df = build_l10_dBdt_time_df(processing, data_key)
    l10_times_1d = l10_gate_times_df.iloc[0].values
    original_data = processing.xyz.layer_data[data_key]

    if method == 'auto':
        use_regression = _needs_regression(l10_times_1d)
    elif method == 'regression':
        use_regression = True
    else:
        use_regression = False

    if use_regression:
        return _regression_slopes(l10_dBdt_df, l10_times_1d)
    else:
        return _adjacent_slopes(l10_dBdt_df, l10_times_1d, original_data)


def calculate_transient_curvatures(processing, data_key, method='auto'):
    """Compute transient decay curvatures in log10(|dBdt|) vs log10(t) space.

    Parameters
    ----------
    processing : object
        Processing object with xyz data and GateTimes.
    data_key : str
        Key for the gate data in layer_data.
    method : str, default 'auto'
        'adjacent'   - 3-point finite difference (original method)
        'regression' - OLS quadratic fit over adaptive windows
        'auto'       - use 'adjacent' if gate spacing is wide, 'regression' if tight

    Returns
    -------
    pd.DataFrame
        Curvature values with same index/columns as the input data.
    """
    l10_dBdt_df, l10_gate_times_df = build_l10_dBdt_time_df(processing, data_key)
    l10_times_1d = l10_gate_times_df.iloc[0].values
    original_data = processing.xyz.layer_data[data_key]

    if method == 'auto':
        use_regression = _needs_regression(l10_times_1d)
    elif method == 'regression':
        use_regression = True
    else:
        use_regression = False

    if use_regression:
        return _regression_curvatures(l10_dBdt_df, l10_times_1d)
    else:
        return _adjacent_curvatures(l10_dBdt_df, l10_times_1d, original_data)


def sampleDEM_reproject_DEM(DEMfilename,
              df,
              poskeys=['utmx', 'utmy'],
              z_key='zdem_laser1',
              crs=None,
              force_overwrite=False):
    xmin = df[poskeys[0]].min()
    xmax = df[poskeys[0]].max()
    ymin = df[poskeys[1]].min()
    ymax = df[poskeys[1]].max()

    coord_list = [(x, y) for x, y in zip(df[poskeys[0]], df[poskeys[1]])]

    dem = rasterio.open(DEMfilename, 'r')
    left, bottom, right, top = dem.bounds

    if crs is None:
        print('Assuming that the DTM projection is the same as the AEM data')
        crs = dem.crs

    if type(crs) is int:
        dst_crs = f"EPSG:{crs}"
    elif 'EPSG:' in str(crs):
        dst_crs = crs
    elif 'epsg:' in str(crs):
        dst_crs = crs.replace('epsg', 'EPSG')

    assert 'dst_crs' in locals(), f'There is something wrong with the supplied crs parameter ({crs}).'
    dst_crs = rasterio.crs.CRS.from_string(dst_crs)

    if dem.crs != crs:
        print(f"  - DTM projection ({dem.crs}) is not the same as specified projection ({dst_crs}).")
        print(f"      Re-projecting the DTM to {dst_crs}.")
        if str(dem.crs).split('EPSG:')[1]+'.tif' in DEMfilename:
            new_DEMfilename = DEMfilename.replace(str(dem.crs).split('EPSG:')[1], str(dst_crs).split('EPSG:')[1])
        else:
            new_DEMfilename = DEMfilename.replace('.tif', f"_{str(dst_crs).split('EPSG:')[1]}.tif")

        if os.path.isfile(new_DEMfilename) and not force_overwrite:
            print(f"\n****   Note: {new_DEMfilename} already exists, using this file   ****")
            reprojected_dem = rasterio.open(new_DEMfilename, 'r')
        else:
            print(f"{new_DEMfilename} does not exist. Writing file now")
            transform, width, height = calculate_default_transform(dem.crs,
                                                                   dst_crs,
                                                                   dem.width,
                                                                   dem.height,
                                                                   *dem.bounds)
            kwargs = dem.meta.copy()
            kwargs.update({'crs': dst_crs,
                           'transform': transform,
                           'width': width,
                           'height': height
                           })
            new_dem = rasterio.open(new_DEMfilename, 'w', **kwargs)
            with rasterio.open(new_DEMfilename, 'w', **kwargs) as new_dem:
                for i in range(1, dem.count + 1):
                    reproject(source=rasterio.band(dem, i),
                              destination=rasterio.band(new_dem, i),
                              src_transform=dem.transform,
                              src_crs=dem.crs,
                              dst_transform=transform,
                              dst_crs=dst_crs,
                              resampling=Resampling.nearest)

            dem.close()
            reprojected_dem = rasterio.open(new_DEMfilename, 'r')
    else:
        reprojected_dem = dem

    repro_dem_xmin, repro_dem_ymin, repro_dem_xmax, repro_dem_ymax = reprojected_dem.bounds

    for a, b in zip([          xmin,           ymin, repro_dem_xmax, repro_dem_ymax],
                    [repro_dem_xmin, repro_dem_ymin,           xmax,           ymax]):
        if a < b:
            bounds_comparison = pd.DataFrame([[left, bottom, right, top],
                                              [repro_dem_xmin, repro_dem_ymin, repro_dem_xmax, repro_dem_ymax],
                                              [xmin, ymin, xmax, ymax]],
                                             index=['orig_dem', 'reprojected_dem', 'flightlines'],
                                             columns=['left', 'top', 'right', 'bottom'])
            print(bounds_comparison)
            raise Exception('coordinates outside raster bounds')

    dtm_values = [x[0] for x in reprojected_dem.sample(coord_list)]
    df[z_key] = dtm_values
    reprojected_dem.close()
    dem.close()

def drop_lines_from_data(data, line_list, line_key='Line'):
    for line in line_list:
        filt=data.flightlines[line_key]==line
        data.flightlines = data.flightlines.drop(data.flightlines.iloc[filt.values].index)
        data.flightlines.reset_index(drop=True, inplace=True)
        for key in data.layer_data.keys():
            data.layer_data[key]=data.layer_data[key].drop(data.layer_data[key].iloc[filt.values].index)
            data.layer_data[key].reset_index(drop=True, inplace=True)

def calcGateRelErr(data, synth):
    print('calculating inversion error gate by gate, sounding by sounding.')
    for moment in allowed_moments:
        if moment in data.layer_data.keys():
            std_key='STD_'+moment.split('Gate_')[-1]
            relErr_key='relErr_'+moment.split('Gate_')[-1]
            err=data.layer_data[moment].abs()*data.layer_data[std_key]
            data.layer_data[relErr_key]=(data.layer_data[moment].abs()-synth.layer_data[moment].abs()) / err


def drop_filt_XYZ(data, filt, reset_index=True):
    data.flightlines=data.flightlines.drop(data.flightlines.loc[filt,:].index)
    if reset_index:
        data.flightlines.reset_index(inplace=True)
        data.flightlines.drop(['index'], axis=1, inplace=True)
    for key in data.layer_data.keys():
        data.layer_data[key]=data.layer_data[key].drop(data.layer_data[key].loc[filt,:].index)
        if reset_index:
            data.layer_data[key].reset_index(inplace=True)
            data.layer_data[key].drop(['index'], axis=1, inplace=True)

def filtXYZ(data, filt, reset_index=True):
    data_out=copy.deepcopy(data)
    drop_filt_XYZ(data_out, ~filt, reset_index=reset_index)
    return data_out


def substractSystemBias(data, System_bias_dict):
    for moment in allowed_moments:
        if moment in System_bias_dict.keys():
            if data.layer_data[moment].shape[1] == len(System_bias_dict[moment]):
                print('Correcting system bias for {}'.format(moment))
                Bias=pd.DataFrame(np.tile(System_bias_dict[moment].values, (len(data.layer_data[moment]), 1)))
                data.layer_data[moment]=data.layer_data[moment]-Bias
            else:
                raise Exception("Number of gates in system bias and data structure differ!") 
        else:
            print('Moment: {} not found in bias dictionary'.format(moment) )


def round_to_odd(f):
    return int(np.ceil(f/2)  * 2 - 1)

def interpolate_rolling_size_for_all_gates(filterlist, moment):
    ci = moment.columns.values.astype(int)
    c = [0, ci.max()]
    f = interp1d(c, filterlist)
    ni = f(ci)
    return [round_to_odd(n) for n in ni]

def get_min_periods(filter_length, min_valid_fraction=0.35):
    """
    Calculate minimum periods as a fraction of filter length.

    Parameters
    ----------
    filter_length : int
        The rolling window size
    min_valid_fraction : float, default 0.35
        Minimum fraction of window that must have valid (non-NaN) data before
        the filter produces an output. See moving_average_filter for guidance
        on choosing this value.

    Returns
    -------
    int
        Minimum number of valid samples required
    """
    if filter_length > 1:
        return max(2, int(np.ceil(filter_length * min_valid_fraction)))
    else:
        return 1


def alpha_trim(data, errors, alpha=0.1):
    """
    Remove top and bottom alpha fraction of data points.

    Parameters
    ----------
    data : array-like
        Data values (NaN already removed)
    errors : array-like
        Error estimates corresponding to data
    alpha : float, default 0.1
        Fraction to trim from each end (0.1 = 10% from each end)

    Returns
    -------
    trimmed_data : ndarray
        Data with extreme values removed
    trimmed_errors : ndarray
        Corresponding errors
    """
    n = len(data)
    if n < 3:  # Can't trim if too few points
        return np.asarray(data), np.asarray(errors)

    n_trim = int(n * alpha)
    if 2 * n_trim >= n:  # Would trim everything
        return np.asarray(data), np.asarray(errors)

    # Sort by data value, trim extremes
    sort_idx = np.argsort(data)
    keep_idx = sort_idx[n_trim:-n_trim] if n_trim > 0 else sort_idx

    return np.asarray(data)[keep_idx], np.asarray(errors)[keep_idx]


def inverse_variance_weights(errors, max_weight_factor=10.0, err_floor=None):
    """
    Calculate inverse variance weights with cap to prevent dominance.

    Parameters
    ----------
    errors : array-like
        Error estimates (standard deviations)
    max_weight_factor : float, default 10.0
        Maximum weight as multiple of median weight.
        Prevents single low-error measurement from dominating.
    err_floor : float or None, default None
        External minimum error floor (e.g. from the full gate column).
        When provided, ensures windows with all-zero errors still get
        a meaningful floor. The effective floor is the max of this and
        the per-window median.

    Returns
    -------
    weights : ndarray
        Weights (not normalized, for use with variance_averaging functions)
    """
    errors = np.asarray(errors, dtype=float)

    # Floor near-zero errors to prevent infinite weights.
    # With fractional error models (abs_err = value * frac_err), near-zero
    # data values get near-zero errors, implying false infinite precision.
    abs_errors = np.abs(errors)
    nonzero = abs_errors[abs_errors > 0]
    window_floor = np.median(nonzero) if len(nonzero) > 0 else 0.0

    # Use the larger of the per-window floor and the external gate-level floor
    effective_floor = max(window_floor, err_floor or 0.0)

    if effective_floor > 0:
        abs_errors = np.maximum(abs_errors, effective_floor)
    else:
        # No error information at all — fall back to equal weights
        return np.ones(len(errors))

    # Inverse variance weighting
    variances = abs_errors ** 2
    weights = 1.0 / variances

    # Cap weights to prevent dominance
    median_weight = np.median(weights)
    max_weight = median_weight * max_weight_factor
    weights = np.minimum(weights, max_weight)

    return weights


def rolling_hybrid_mean_df(df_dat, df_err_fp, rolling_lengths,
                           alpha=0.1, min_valid_fraction=0.35, max_weight_factor=10.0):
    """
    Rolling average with hybrid alpha-trim + inverse variance weighting.

    This method:
    1. Excludes NaN values (culled data)
    2. Alpha trims extreme values for robustness
    3. Applies inverse variance weighting for optimal estimation
    4. Computes combined uncertainty using SST formula with weights

    Parameters
    ----------
    df_dat : DataFrame
        Data values
    df_err_fp : DataFrame
        Fractional error estimates (STD / value)
    rolling_lengths : list
        Window size for each column (gate)
    alpha : float, default 0.1
        Fraction to trim from each end (0.1 = 10%)
    min_valid_fraction : float, default 0.35
        Minimum fraction of window needed for valid output
    max_weight_factor : float, default 10.0
        Cap for inverse variance weights

    Returns
    -------
    ave_dat : DataFrame
        Weighted averaged data
    frac_err : DataFrame
        Fractional error of averaged data
    """
    if len(rolling_lengths) != len(df_dat.columns):
        raise ValueError(f'Number of rolling filter lengths ({len(rolling_lengths)}) '
                        f'differs from number of columns ({len(df_dat.columns)})')

    index_shift = min(df_dat.index)

    # Calculate absolute errors
    df_err_ab = df_dat * df_err_fp

    # Pre-compute per-gate error floor from the full line context.
    # This prevents all-zero windows (common in late gates at the noise floor)
    # from producing NaN via the inf*0 path. The gate-level floor reflects
    # the typical error magnitude across the entire line for that gate.
    gate_err_floors = {}
    for col in df_dat.columns:
        col_abs_err = np.abs(df_err_ab[col].dropna())
        nonzero = col_abs_err[col_abs_err > 0]
        gate_err_floors[col] = np.median(nonzero) if len(nonzero) > 0 else 0.0

    # Prepare output dataframes
    ave_dat = df_dat * np.nan
    std_err_ab = df_err_ab * np.nan

    for filter_length, col in zip(rolling_lengths, df_dat.columns):
        min_periods = get_min_periods(filter_length, min_valid_fraction)

        for sid in range(len(df_dat[col])):
            # Define window bounds
            half_window = int(np.floor(filter_length / 2))
            win_start = max(0, sid - half_window)
            win_end = min(len(df_dat[col]), sid + half_window + 1)

            # Extract window data
            window_data = df_dat[col].loc[win_start + index_shift: win_end - 1 + index_shift]
            window_err = df_err_ab[col].loc[win_start + index_shift: win_end - 1 + index_shift]

            # Filter NaN values
            valid_mask = window_data.notna() & window_err.notna()
            valid_data = window_data[valid_mask].values
            valid_err = window_err[valid_mask].values

            if len(valid_data) < min_periods:
                continue  # Not enough valid samples

            # Alpha trim
            trimmed_data, trimmed_err = alpha_trim(valid_data, valid_err, alpha)

            if len(trimmed_data) < 2:
                continue  # Need at least 2 for variance

            # Calculate inverse variance weights with gate-level error floor
            weights = inverse_variance_weights(trimmed_err, max_weight_factor,
                                               err_floor=gate_err_floors[col])

            # Weighted mean
            weighted_mean = np.sum(weights * trimmed_data) / np.sum(weights)

            # Guard against floating-point cancellation artifacts.
            # When window values nearly cancel (e.g. [0.3, -0.3, 0.1, -0.1]),
            # both IVW weighted mean and simple mean can produce tiny ghost
            # values (~1e-18) from floating-point residuals. If the result
            # is >12 orders of magnitude smaller than the data, it's an
            # artifact — the true average is effectively zero.
            mean_abs = np.mean(np.abs(trimmed_data))
            if mean_abs > 0 and abs(weighted_mean) / mean_abs < 1e-12:
                weighted_mean = 0.0

            # Calculate variance using SST formula with weights
            var_est = variance_averaging.calcVarSST(
                n=weights,
                mu=trimmed_data,
                sd=trimmed_err,
                mu_tot=weighted_mean
            )

            # Store results
            ave_dat.loc[sid + index_shift, col] = weighted_mean
            std_err_ab.loc[sid + index_shift, col] = np.sqrt(var_est)

    # Convert to fractional error
    frac_err = np.abs(std_err_ab / ave_dat)

    return ave_dat, frac_err


def deprecated_rolling_weighted_mean_df(df_dat, df_err_fp, rolling_lengths, weighting_factor=3, error_calc_scheme='Weighted_SEM'):
    assert weighting_factor > 0, "weighting_factor must be greater than 0. Suggested ranges are between 1 [Weights are only based on the errors - errors will be smaller] and 10 [errors will be bigger]"
    if len(rolling_lengths) == len(df_dat.columns):
        # Calculate absolute errors
        df_err_ab = df_dat * df_err_fp

        # Calculate weights
        # FIXME: I'm applying a factor of 3 here. This is purely determined experimentally. Since this is just a
        #   weighting function I think it's ok?
        weights_df = 1 / (weighting_factor * (df_err_ab**2))

        # Build weighted data df for the averaging.
        weighted_data = df_dat * weights_df

        # Prepare empty data frames
        ave_dat = df_dat * np.nan
        ave_err_abs_df = df_err_ab * np.nan
        std_err_df = df_err_ab * np.nan
        unweighted_SEM_df = df_err_ab * np.nan
        weighted_SEM_df = df_err_ab * np.nan

        for filter_length, col in zip(rolling_lengths, df_dat.columns):
            # Calculate the rolling mean of the absolute error
            ave_err_abs_df[col] = df_err_ab[col].rolling(filter_length, center=True, min_periods=get_min_periods(filter_length)).mean()

            # Calculate the rolling STD error
            std_err_df[col] = df_dat[col].rolling(filter_length, center=True, min_periods=get_min_periods(filter_length)).std()

            # Calculate the unweighted Standard Error of the Mean
            unweighted_SEM_df[col] = df_dat[col].rolling(filter_length, center=True, min_periods=get_min_periods(filter_length)).std() / np.sqrt(filter_length)

            # Calculate the weighted average of the data
            ave_dat[col] = weighted_data[col].rolling(filter_length, center=True, min_periods=get_min_periods(filter_length)).sum() / \
                              weights_df[col].rolling(filter_length, center=True, min_periods=get_min_periods(filter_length)).sum()

            # Calculate the weighted Standard Error of the Mean
            weighted_SEM_df[col] = (weights_df[col].rolling(filter_length, center=True, min_periods=get_min_periods(filter_length)).sum() /
                                   (weights_df[col].rolling(filter_length, center=True, min_periods=get_min_periods(filter_length)).sum()**2))**(1/2)

        # Balance absolute errors by the 1) the mean, 2) the STD, and 3) the weighted SEM 4) unweighted SEM
        # FIXME: I have a hard time to justify this balancing, but without it I feel that the errors from the SEM alone are too small
        unweighted_SEM_weight = 1
        weighted_SEM_weight = 1
        STD_weight = 1
        mean_weight = 1

        divide_by = unweighted_SEM_weight + weighted_SEM_weight + STD_weight + mean_weight

        balanced_abs_err1 = (unweighted_SEM_df * weighted_SEM_df * std_err_df * ave_err_abs_df) ** (1 / divide_by)

        balanced_abs_err2 = (unweighted_SEM_weight * (unweighted_SEM_df**2) / divide_by +
                             weighted_SEM_weight   * (weighted_SEM_df**2)   / divide_by +
                             STD_weight            * (std_err_df**2)        / divide_by +
                             mean_weight           * (ave_err_abs_df**2)    / divide_by  )**(1/2)

        # calculate the fractional error of the balanced absolute error
        weighted_SEM_frac_err =     np.abs(weighted_SEM_df / ave_dat)
        unweighted_SEM_frac_err = np.abs(unweighted_SEM_df / ave_dat)
        std_frac_err =                   np.abs(std_err_df / ave_dat)
        ave_frac_err =               np.abs(ave_err_abs_df / ave_dat)
        balanced_frac_err1 =      np.abs(balanced_abs_err1 / ave_dat)
        balanced_frac_err2 =      np.abs(balanced_abs_err2 / ave_dat)

        if error_calc_scheme == 'Weighted_SEM':
            return ave_dat, weighted_SEM_frac_err
        elif error_calc_scheme == 'Balanced_1':
            return ave_dat, balanced_frac_err1
        elif error_calc_scheme == 'Average':
            return ave_dat, ave_frac_err
        elif error_calc_scheme == 'Balanced_2':
            return ave_dat, balanced_frac_err2
        elif error_calc_scheme == 'STD':
            return ave_dat, std_frac_err
        elif error_calc_scheme == 'Unweighted_SEM':
            return ave_dat, unweighted_SEM_frac_err

    else:
        print(f'filter length: {len(rolling_lengths)}')
        print(f'number of data columns: {len(df_dat.columns)}')
        print(f'number of std columns: {len(df_err_fp.columns)}')
        raise Exception('number of rolling filter lengths differs from number of columns in dataframe ')


def rolling_SST_mean_df(df_dat, df_err_fp, rolling_lengths, min_valid_fraction=0.35):
    if len(rolling_lengths) == len(df_dat.columns):
        index_shift = min(df_dat.index)

        # Calculate absolute errors
        df_err_ab = df_dat * df_err_fp

        # Prepare empty data frames
        ave_dat = df_dat * np.nan
        std_SST_err_ab = df_err_ab * np.nan

        for filter_length, col in zip(rolling_lengths, df_dat.columns):
            # Calculate the average of the data
            ave_dat[col] = df_dat[col].rolling(filter_length, center=True, min_periods=get_min_periods(filter_length, min_valid_fraction)).mean()

            # Calculate the SST error
            for sid in range(0, len(df_dat[col])):
                current_window = [int(sid - np.floor(filter_length / 2)), int(sid + np.floor(filter_length / 2) + 1)]
                if current_window[0] < 0:
                    current_window[0] = 0
                if current_window[1] > len(df_dat[col]):
                    current_window[1] = len(df_dat[col])
                # Extract window data and errors
                window_data = df_dat[col].loc[current_window[0] + index_shift: current_window[1] - 1 + index_shift]
                window_std = df_err_ab[col].loc[current_window[0] + index_shift: current_window[1] - 1 + index_shift]

                # Create mask for valid (non-NaN) values in both data and error
                valid_mask = window_data.notna() & window_std.notna()
                num_sample = valid_mask.sum()

                if num_sample >= get_min_periods(filter_length, min_valid_fraction):
                    sample_data = window_data[valid_mask].values
                    sample_std = window_std[valid_mask].values
                    sample_weight = np.ones(num_sample)
                    estimatedSST = ave_dat[col].loc[current_window[0] + index_shift: current_window[1] - 1 + index_shift][valid_mask].mean()

                    var_est_SST = variance_averaging.calcVarSST(n=sample_weight,
                                                                mu=sample_data,
                                                                sd=sample_std,
                                                                mu_tot=estimatedSST)
                    std_est_SST = var_est_SST ** 0.5
                    std_SST_err_ab.loc[sid + index_shift, col] = std_est_SST

        SST_frac_err = np.abs(std_SST_err_ab / ave_dat)

        return ave_dat, SST_frac_err
    else:
        print(f'filter length: {len(rolling_lengths)}')
        print(f'number of data columns: {len(df_dat.columns)}')
        print(f'number of std columns: {len(df_err_fp.columns)}')
        raise Exception('number of rolling filter lengths differs from number of columns in dataframe ')


def rolling_mean_df(df_dat, rolling_lengths, error_calc_scheme='Unweighted_SEM',
                    min_valid_fraction=0.35):
    if len(rolling_lengths) == len(df_dat.columns):
        # Prepare empty data frames
        ave_dat = df_dat * np.nan
        std_err_df = df_dat * np.nan
        unweighted_SEM_df = df_dat * np.nan

        for filter_length, col in zip(rolling_lengths, df_dat.columns):
            min_periods = get_min_periods(filter_length, min_valid_fraction)
            # Calculate the rolling STD error
            std_err_df[col] = df_dat[col].rolling(filter_length, center=True, min_periods=min_periods).std()

            # Calculate the unweighted Standard Error of the Mean
            rolling_std = df_dat[col].rolling(filter_length, center=True, min_periods=min_periods).std()
            actual_count = df_dat[col].rolling(filter_length, center=True, min_periods=min_periods).count()
            unweighted_SEM_df[col] = rolling_std / np.sqrt(actual_count)

            ave_dat[col] = df_dat[col].rolling(filter_length, center=True, min_periods=min_periods).mean()

        unweighted_SEM_frac_err = np.abs(unweighted_SEM_df / ave_dat)
        std_frac_err = np.abs(std_err_df / ave_dat)

        if error_calc_scheme == 'Unweighted_SEM':
            return ave_dat, unweighted_SEM_frac_err
        elif error_calc_scheme == 'STD':
            return ave_dat, std_frac_err

    else:
        print(f'filter length: {len(rolling_lengths)}')
        print(f'number of columns: {len(df_dat.columns)}')
        raise Exception('number of rolling filter lengths differs from number of columns in dataframe ')


def rolling_square_root_sum_df(df, rolling_lengths):
    if len(rolling_lengths) == len(df.columns):
        df_out = copy.deepcopy(df)
        for filter_length, col in zip(rolling_lengths, df_out.columns):
            notna_length = df_out[col].notna().rolling(filter_length, center=True, min_periods=1).sum()
            # df_out[col] = np.sqrt(1/notna_length**2 * (df_out[col]*df_out[col]).rolling(filter_length, center=True, min_periods=get_min_periods(filter_length)).sum())
            df_out[col] = np.sqrt(1/notna_length * (df_out[col] * df_out[col]).rolling(filter_length, center=True, min_periods=get_min_periods(filter_length)).sum())
    else:
        # print(f'Filter length: {len(rolling_lengths)}')
        # print(f'Number of columns: {len(df.columns)}')
        raise Exception(f'Number of rolling filter lengths ({len(rolling_lengths)}) differs from number of columns in dataframe ({len(df.columns)})')
    return df_out


def make_noise_df(processing,
                  channel=1,
                  noise_level_1ms=1e-8,
                  noise_exponent=-0.5,
                  norm_by_tx: bool = True):
    """


    Parameters
    ----------
    channel :
        Which channel to use
    noise_level_1ms :
        amplitude of the noise floor, in V/m^2
    noise_exponent :
        Slope of the noise floor, in d(log10(V/m^2))/d(s), equivalent to t^slope
    norm_by_tx :
        Normalize by transmitter? If:
        True - the output unit is V/(A*m^4) [normalized by transmitter and receiver]
        False - the output unit is V/(m^2) [normalized by the receiver only]

    Returns
    -------
    noise_df : Pandas Dataframe
        A dataframe, the same shape as the data, with amplitudes representing the noise floor.
        if 'norm_by_tx' is False the output will be in V/(m^2), or normalized by the rx only.
        If 'norm_by_tx' is True (default) the output will be in V/(A*m^4), or normalized by the rx and transmitter moment
    """

    # build a dataframe that holds the noise levels for the individual gates:
    data = processing.xyz
    num_soundings = len(data.flightlines.index)

    str_channel = f"0{channel}"[-2:]
    data_key = f"{dat_key_prefix}{str_channel}"
    assert data_key in data.layer_data.keys(), "The channel requested does not exist"

    gate_times = np.array(processing.GateTimes[data_key], dtype=float)
    noise = (np.clip(gate_times, 0, None) / 1e-3) ** noise_exponent * noise_level_1ms

    noise_array = np.tile(noise, (num_soundings, 1))
    noise_df = pd.DataFrame(noise_array,
                            dtype=float,
                            columns=data.layer_data[data_key].columns,
                            index=data.layer_data[data_key].index)

    noise_df = noise_df / data.model_info.get('scalefactor', 1.0)
    if norm_by_tx:
        noise_df = noise_df / processing.ApproxDipoleMoment[data_key]

    return noise_df


# def make_noise_dict(processing, channel=1, noise_level_1ms=1e8, noise_exponent=-0.5, unitDipole=False):
#     # build a dictionary that holds the noise levels for the individual gates:
#     data = processing.xyz
#
#     str_channel = f"0{channel}"[-2:]
#     data_key = f"{dat_key_prefix}{str_channel}"
#     assert data_key in data.layer_data.keys(), "The channel requested does not exist"
#
#     noise_dict = {}
#     noise = ((processing.GateTimes[data_key] * 1e3)**noise_exponent) * noise_level_1ms
#     noise_dict[data_key] = {}
#     for k, column in enumerate(data.layer_data[data_key].columns):
#         noise_dict[data_key][column] = noise[k]
#     if unitDipole:
#         for data_key in noise_dict.keys():
#             for key in noise_dict[data_key].keys():
#                 noise_dict[data_key][key] = noise_dict[data_key][key] / data.model_info['scalefactor'] / processing.ApproxDipoleMoment[data_key]
#
#     return noise_dict
#
# def add_noise_dict(processing, channel, noise_dict):
#     data = processing.xyz
#     str_channel = f"0{channel}"[-2:]
#     data_key = f"{dat_key_prefix}{str_channel}"
#
#     assert data_key in data.layer_data.keys(), "The channel requested does not exist"
#     assert 'scalefactor' in data.model_info.keys(), "data.model_info['scalefactor'] must be defined"
#
#     processing.Noise = {}
#
#     for key in noise_dict[data_key].keys():
#         processing.Noise[data_key] = np.array(list(noise_dict[data_key].values()))
                
def sortXYZ(data, sort_by_column):
    xyz=copy.deepcopy(data)
    for key in xyz.layer_data.keys():
        if sum(xyz.flightlines.index == xyz.layer_data[key].index)<len(xyz.flightlines.index):
            raise Exception('Indexes from flightlines and layer_data differ!')
    xyz.flightlines.sort_values(sort_by_column, inplace=True)
    for key in xyz.layer_data.keys():
        xyz.layer_data[key]=xyz.layer_data[key].loc[xyz.flightlines.index,:]
    return xyz

    

def scale_to_picoVolt(xyz):
    print("Warning: Ignoring existing scaling. This might give wrong results!")
    for moment in allowed_moments:
        xyz.layer_data[moment] = xyz.layer_data[moment]*1e12
        xyz.model_info['scalefactor']=1e-12

def add_inuse_flags(xyz, gex=None):
    for moment in allowed_moments:
        if moment in xyz.layer_data.keys():
            inuse_key=inuse_moment(moment)
            xyz.layer_data[inuse_key]=pd.DataFrame(np.ones(xyz.layer_data[moment].loc[:,:].shape).astype(int)).set_index(xyz.flightlines.index, drop=True)
            if gex:
                if type(gex)==str:
                    gex=libaarhusxyz.gex.parse(gex)
                gex_ch_key='Channel'+moment[-1]
                for col in range(int(gex[gex_ch_key]['RemoveInitialGates'])):
                    xyz.layer_data[inuse_key].loc[:,col]=0

def remove_empty_soundings(data):
    filt=pd.Series(np.ones(len(data.flightlines))).astype(bool)
    for moment in allowed_moments:
        if moment in data.layer_data.keys():
            no_data = data.layer_data[inuse_moment(moment)].sum(axis=1)==0
            print('{0} has {1} sounding positions without data'.format(moment, no_data.sum()))
            filt=filt & no_data
    drop_filt_XYZ(data, filt)

