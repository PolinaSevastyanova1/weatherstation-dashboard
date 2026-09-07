import json
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import xarray as xr
from pathlib import Path

# defining data file
DATA_FILE_15 = os.path.join("..", "data", "L0", "CR1000_1436_Folgefonna_Modem_Table15min.dat")
DATA_FILE_1 = os.path.join("..", "data", "L0", "CR1000_1436_Folgefonna_Modem_Table1min.dat")

# method for loading in data file
def load_toa5_table(path):
    names = pd.read_csv(path, nrows=0, header=1).columns
    df    = pd.read_csv(path, header=0, names=names, skiprows=3, na_values=['NAN'])
    return df

# loading data and renaming columns
df15 = load_toa5_table(DATA_FILE_15)
df1  = load_toa5_table(DATA_FILE_1)

# Giving the columns more clear names
df15.columns = ['LoggerTime', 'Record', 'ID', 'BattMin', 'BattMinTime', 'PanelTemp',
'Pressure', 'TiltX', 'TiltY', 'LatitudeA', 'LatitudeB', 'LongitudeA',
'LongitudeB', 'MagVar', 'NumSats', 'Altitude', 'FixQual', 
'MaxClockChange', 'NumClockChange', 'PPS', 'SecSinceGPRMC',
'NMEASent(1)', 'NMEASent(2)', 'GNSSTime', 'SnowHeight',
'SignalQuality', 'Temp1', 'Temp2', 'Hum1', 'Hum1Temp', 'Hum2',
'Hum2Temp']
df1.columns = ['LoggerTime', 'Record', 'ID', 'WindSpeed', 'WindDir', 'WindDirSTDEV', 'WindMax',
'WindSpeedSTDEV', 'F2_Mean', 'F_2_WVc(2)', 'F_2_WVc(3)', 'F2_Max', 'F2_stdev',
'SWin', 'SWout', 'LWin', 'LWout', 'CNR4Temp', 'LWinUCORR',
'LWoutUCORR', 'Precip', 'SnowHeight', 'SignalQuality']

# converting date columns to datetime format
df15['LoggerTime'] = pd.to_datetime(df15['LoggerTime'], errors='coerce')
df1['LoggerTime'] = pd.to_datetime(df1['LoggerTime'], errors='coerce')
df15['BattMinTime'] = pd.to_datetime(df15['BattMinTime'], errors='coerce')

# Getting GNSS date and time from gprmc string
def parse_gnss_time(row):
    gprmc = row["NMEASent(1)"].split(",")
    (hhmmss, ddmmyy) = (gprmc[1], gprmc[9])
    return pd.Timestamp(year=2000 + int(ddmmyy[4:6]), month=int(ddmmyy[2:4]), day=int(ddmmyy[0:2]), hour=int(hhmmss[0:2]), minute=int(hhmmss[2:4]), second=int(float(hhmmss[4:6])))
# getting lat, long, altitude etc from gpgga string, converting to decimal format from NMEA format
def parse_location(row):
    gpgga = row["NMEASent(2)"].split(",")
    lat = float(gpgga[2]); lat_deg = int(lat/100); lat_min = lat%100; lat_dd = lat_deg + (lat_min / 60);
    lon = float(gpgga[4]); lon_deg = int(lon/100); lon_min = lon%100; lon_dd = lon_deg + (lon_min / 60);
    return pd.Series({'Latitude': round(lat_dd, 6), 'Longitude': round(lon_dd, 6), 'Altitude': gpgga[9], 'HDOP': float(gpgga[8]), 'GeoidSeparation': gpgga[11]})
df15['GNSSTime'] = df15.apply(parse_gnss_time, axis=1)
df15[['Latitude', 'Longitude', 'Altitude', 'HDOP', 'GeoidSeparation']] = df15.apply(parse_location, axis=1)

# calculating time lag between logger and gnss time
df15["LoggerTimeDiff"] = (df15['LoggerTime'] - df15['GNSSTime']).dt.total_seconds()
print(f"[LOG] Final Logger Time Difference: {df15['LoggerTimeDiff'][len(df15)-1]} seconds")

# Filtering GPS data by fix quality, must be more than plain GPS fix
initial_count = len(df15)
df15 = (df15[df15['Altitude'].notna()
             & (df15['HDOP'].notna())
             & (df15['FixQual'] >= 2)
             & (df15['NumSats'] >= 5)
             & (df15['HDOP'] <= 2.0)])
print(f"[LOG] GPS Quality Filter: Removed {initial_count - len(df15)} low-quality fix data points from {initial_count} total records ({( (initial_count - len(df15)) / initial_count * 100 ):.1f}%)") 

# Removing columns that are not needed
df15 = df15.drop(columns=['PPS', 'MaxClockChange', 'NumClockChange', 'LatitudeA', 'LatitudeB', 'LongitudeA',
'LongitudeB', 'NMEASent(1)', 'NMEASent(2)', 'SnowHeight', 'SignalQuality'])
df1 = df1.drop(columns=['SnowHeight', 'SignalQuality','F_2_WVc(2)','F_2_WVc(3)','F2_Mean','F2_stdev','F2_Max', 'LWinUCORR', 'LWoutUCORR'])
print('Dropped redundant columns')

# INITIAL SUMMARY STATISTICS
# ── Summary statistics table ──────────────────────────────────────────────────
cols = ['Temp1','Temp2','Hum1','Hum2','Hum1Temp','Hum2Temp',
        'WindSpeed','WindDir','WindMax','WindSpeedSTDEV','WindDirSTDEV','Pressure','Precip',
        'SWin','SWout','LWin','LWout', 
        'CNR4Temp', 'BattMin', 'Altitude', 'TiltX', 'TiltY']
# Compute summary statistics using the original (unchanged) df15 and df1

summary = []
for c in cols:
    if c in df15.columns:
        ser = pd.to_numeric(df15[c], errors='coerce')
    elif c in df1.columns:
        ser = pd.to_numeric(df1[c], errors='coerce')
    else:
        ser = pd.Series(dtype=float)
    mn = ser.min()
    mx = ser.max()
    mean = ser.mean()
    three_std_plus = mean + 1.5 * ser.std()
    three_std_minus = mean - 1.5 * ser.std()
    na_count = int(ser.isna().sum())
    summary.append({'Parameter': c, 'Min': None if pd.isna(mn) else round(float(mn),3),
                    'Max': None if pd.isna(mx) else round(float(mx),3),
                    'Mean': None if pd.isna(mean) else round(float(mean),3),
                    '3StdPlus': None if pd.isna(three_std_plus) else round(float(three_std_plus),3),
                    '3StdMinus': None if pd.isna(three_std_minus) else round(float(three_std_minus),3),
                    'NaNCount': na_count})

summary_df = pd.DataFrame(summary).set_index('Parameter')
print('\n[SUMMARY] Statistics (min, max, mean, 3StdPlus, 3StdMinus, NaNCount):')
print(summary_df.to_string())
print()

#CALIBRATION CORRECTIONS

#temp, hum, wind. based on my calibration data.
# FILL THIS WITH CORRECT VALUES
temp_true = np.array([-5,0,5])
hum_true = np.array([-5,0,5])
temp_measured_1 = np.array([-5,0,5])
temp_measured_2=np.array([-5,0,5])
hum_measured_1 = np.array([-5,0,5])
hum_measured_2=np.array([-5,0,5])
m1,c1=np.polyfit(temp_measured_1, temp_true, 1)
m2,c2=np.polyfit(temp_measured_2,temp_true, 1)
m3,c3 = np.polyfit(hum_measured_1,temp_true, 1)
m4,c4 = np.polyfit(hum_measured_2,temp_true, 1)
# Printing the slopes and intercepts
print(f"[CALIB] \n Temp1: slope={m1:.4f}, intercept={c1:.4f} \n Temp2: slope={m2:.4f}, intercept={c2:.4f} \n Hum1: slope={m3:.4f}, intercept={c3:.4f} \n Hum2: slope={m4:.4f}, intercept={c4:.4f}")

df15['Temp1'] = m1 * df15['Temp1'] + c1
df15['Temp2'] = m2 *df15['Temp2'] + c2
df15['Hum1'] = m3 *df15['Hum1'] + c3
df15['Hum2'] = m4 *df15['Hum2'] + c4

# CALIBRATION CORRECTIONS END

#Clipping negative shortwave radiation values to be zero if negative and reporting number of values
clip = 0
for col in ['SWin', 'SWout']:
    clipped_count = int((df1[col] < clip).sum())
    total_count = int(df1[col].notna().sum())
    df1[col] = df1[col].clip(lower=clip)
    pct_clipped = (clipped_count / total_count * 100.0)
    print(f"[LOG] {col} clipped {clipped_count} values to {clip} ({pct_clipped:.1f}%)")
#Clipping humidity to be within 0 and 100 to avoid erasing data
for col in ['Hum1', 'Hum2']:
    clipped_count = int((df15[col] < clip).sum())
    total_count = int(df15[col].notna().sum())
    df15[col] = df15[col].clip(lower=clip, upper=100)
    pct_clipped = (clipped_count / total_count * 100.0)
    print(f"[LOG] {col} clipped {clipped_count} values to 0 ({pct_clipped:.1f}%)")

# Set WindDir to NaN when WindSpeed is 0
df1.loc[df1['WindSpeed'] <=3, 'WindDir'] = np.nan

#FILTERING FOR OUTLIERS AND CLEANING
# ── Apply quality thresholds (values outside ranges set to NaN)
THRESHOLDS = {
    'Temp1': (-40.0, 30.0),
    'Temp2': (-40.0, 30.0),
    'Hum1':  (0.0, 100.0),
    'Hum2':  (0.0, 100.0),
    'Hum1Temp': (-40.0, 30.0),
    'Hum2Temp': (-40.0, 30.0),
    'WindSpeed': (0.0, 80.0),
    'WindDir': (0.0, 360.0),
    'Pressure': (300.0, 1100.0),
    'Precip': (0.0, 100.0),
    'SWin': (0.0, 2000.0),
    'SWout': (0.0, 2000.0),
    'LWin': (0.0, 1000.0),  
    'LWout': (0.0, 1000.0),
    'Altitude': (1200.0, 2000.0),
    'TiltX': (-25.0, 25.0),
    'TiltY': (-25.0, 25.0),
}
# applying min max thresholds and reporting values removed
def apply_thresholds(df, thresholds, label=None):
    removed = {}
    for col, (lo, hi) in thresholds.items():
        if col in df.columns:
            series = pd.to_numeric(df[col], errors='coerce')
            df[col] = series.where(series.between(lo, hi), other=np.nan)
            removed[col] = int(series.notna().sum() - df[col].notna().sum())
    total = sum(removed.values())
    details = ', '.join(f"{col}={cnt}" for col, cnt in removed.items() if cnt)
    print(f"[LOG] {label} threshold filter removed {total} values ({details})")
    return df
# Apply to both tables
df15 = apply_thresholds(df15, THRESHOLDS, label='15-min')
df1 = apply_thresholds(df1, THRESHOLDS, label='1-min')

# Print number of NaN values in each dataset
nan_count_df15 = int(df15.isna().sum().sum())
nan_count_df1 = int(df1.isna().sum().sum())
print(f"[LOG] NaN values: df15={nan_count_df15}, df1={nan_count_df1}, total={nan_count_df15 + nan_count_df1}")

# Check for duplicate timestamps
df15_dupes = df15['LoggerTime'].duplicated().sum()
df1_dupes = df1['LoggerTime'].duplicated().sum()
print(f"[LOG] Duplicate LoggerTime values: df15={df15_dupes}, df1={df1_dupes}")
if df15_dupes > 0:
    print(f"[LOG] Duplicate timestamps in df15: {df15[df15['LoggerTime'].duplicated(keep=False)]['LoggerTime'].unique()}")
if df1_dupes > 0:
    print(f"[LOG] Duplicate timestamps in df1: {df1[df1['LoggerTime'].duplicated(keep=False)]['LoggerTime'].unique()}")


#SAVING CLEANED DATASETS
#Save processed L1 data for further analysis.
L1_DATA_DIR = Path(__file__).parent.parent / 'data' / 'L1'
L1_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Ensure saved netCDF datasets preserve column names as data variables.
ds1 = xr.Dataset.from_dataframe(df1.set_index('LoggerTime'))
ds15 = xr.Dataset.from_dataframe(df15.set_index('LoggerTime'))

ds1.to_netcdf(L1_DATA_DIR / 'DataTable1.nc')
ds15.to_netcdf(L1_DATA_DIR / 'DataTable15.nc')

df1.to_csv(L1_DATA_DIR / 'DataTable1.dat', index=False, sep=',', header=True, columns=list(df1.columns))
df15.to_csv(L1_DATA_DIR / 'DataTable15.dat', index=False, sep=',', header=True, columns=list(df15.columns))
print(f"[LOG] Saved L1 1-minute data to: {L1_DATA_DIR / 'DataTable1.nc'} and  {L1_DATA_DIR / 'DataTable1.dat'}")
print(f"[LOG] Saved L1 15-minute data to: {L1_DATA_DIR / 'DataTable15.nc'} and  {L1_DATA_DIR / 'DataTable15.dat'}")




# ##THIS IS AFTER ALL CALUCATIONS AND CORRECTIONS
# ## ALL THIS BELOW NEEDS TO GO INTO L2TODASHBOARD BUT THIS
# ## ALSO MEANS CHANGING THE CRON JOBS AND INDEX.HTML DEPENDENCY.
# #RESAMPLING AND MERGING DATA
# # Resample 1-min data to 15-min intervals
# df1_15 = (df1.set_index('LoggerTime')
#              .resample('15min')
#              .agg(
#                  WindSpeed=('WindSpeed', 'mean'),
#                  WindDir=('WindDir', 'mean'),
#                  WindDirVariance=('WindDir', 'std'),
#                  SWin=('SWin', 'mean'),
#                  SWout=('SWout', 'mean'),
#                  LWin=('LWin', 'mean'),
#                  LWout=('LWout', 'mean'),
#                  Precip=('Precip', 'sum'),
#                  WindMax=('WindMax', 'max'),
#                  CNR4Temp=('CNR4Temp', 'mean')
#              )
#              .reset_index())
# # Merge resampled 1-min data with 15-min data, keeping df15 logger times and record numbers.
# df = pd.merge_asof(
#     df15.sort_values('LoggerTime'),
#     df1_15.sort_values('LoggerTime'),
#     on='LoggerTime', direction='nearest', tolerance=pd.Timedelta('5min')
# )
# # Rounding to the correct number of significant figures, as measured by the instruments
# import math
# sigcols = ['WindSpeed','WindDir','WindDirVariance','SWin','SWout','LWin','LWout','Precip','WindMax','CNR4Temp']
# radiometer_cols = {'SWin', 'SWout', 'LWin', 'LWout'}
# for col in sigcols:
#     sig = 4 if col in radiometer_cols else 3
#     df[col] = df[col].apply(
#         lambda x: x if x == 0 or pd.isna(x) else round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)
#     )

# #PRINTING PROCESSED SUMMARY STATISTICS
# # Compute summary statistics for all columns in df.
# # Define string columns to report first with their first non-null value
# stringcols = ['LoggerTimeDiff', 'Longitude', 'Latitude', 'SecSinceGPRMC', 'GNSSTime',
#               'BattMinTime', 'ID', 'LoggerTime', 'Record', 'GeoidSeparation']

# summary = []
# # First, add the stringcols in the requested order with their first value
# for c in stringcols:
#         ser = df[c]
#         first = ser.dropna().iloc[-1] if not ser.dropna().empty else ''
#         summary.append({'Parameter': c, 'First': first, 'Min': '', 'Mean': '', 'Max': ''})

# # Then, for all other columns, report numeric summary (min, mean, max)
# for c in df.columns:
#     if c not in stringcols:
#         num = pd.to_numeric(df[c], errors='coerce')
#         summary.append({
#             'Parameter': c,
#             'Min': None if pd.isna(num.min()) else num.min(),
#             'Mean': None if pd.isna(num.mean()) else num.mean(),
#             'Max': None if pd.isna(num.max()) else num.max(),
#             'First': ''
#         })
# summary_df = pd.DataFrame(summary).set_index('Parameter')
# print('\n[SUMMARY] Statistics (numeric: min, mean, max; non-numeric: first entry):')
# print(summary_df.to_string())
# print()

# #SAVING CLEANED DATASET
# #Save processed L1 data for further analysis.
# L1_DATA_DIR = Path(__file__).parent.parent / 'data' / 'L1'
# L1_DATA_DIR.mkdir(parents=True, exist_ok=True)
# df.set_index('LoggerTime').to_xarray().to_netcdf(L1_DATA_DIR / 'DataTable.nc')
# df.to_csv(L1_DATA_DIR / 'DataTable.dat', index=False, sep='\t')
# print(f"[LOG] Saved L1 data to: {L1_DATA_DIR / 'DataTable.nc'} and  {L1_DATA_DIR / 'DataTable.dat'}")


# #DASHBOARD EXPORT
# # ── Dashboard export ──────────────────────────────────────────────────────────
# # Resample 1-min data to 15-min averages, merge with 15-min table,
# # filter to the last 7 days, and write data.json for the website.

# DASHBOARD_DIR = Path(__file__).parent.parent / 'dashboard'

# last_record = df1['LoggerTime'].max()
# day_start = last_record.normalize()           # midnight at start of that day
# day_end   = day_start + pd.Timedelta('1 day') # midnight at end of that day

# df_met = df15[['LoggerTime', 'Temp1', 'Hum1',
#                'Pressure']].sort_values('LoggerTime')

# df_dash = pd.merge_asof(
#     df1_15.sort_values('LoggerTime'),
#     df_met,
#     on='LoggerTime', direction='nearest', tolerance=pd.Timedelta('8min')
# )

# df_dash = df_dash[(df_dash['LoggerTime'] >= day_start) &
#                   (df_dash['LoggerTime'] <  day_end)].reset_index(drop=True)

# def to_list(series):
#     return [None if pd.isna(v) else round(float(v), 3) for v in series]

# data = {
#     'updated':    pd.Timestamp.now().isoformat(),
#     'day_start':  day_start.isoformat(),
#     'day_end':    day_end.isoformat(),
#     't':          [ts.isoformat() for ts in df_dash['LoggerTime']],
#     'Temp1':      to_list(df_dash['Temp1']),
#     'Hum1':       to_list(df_dash['Hum1']),
#     'WindSpeed':  to_list(df_dash['WindSpeed']),
#     'WindDir':    to_list(df_dash['WindDir']),
#     'WindMax':    to_list(df_dash['WindMax']),
#     'SWin':       to_list(df_dash['SWin']),
#     'SWout':      to_list(df_dash['SWout']),
#     'LWin':       to_list(df_dash['LWin']),
#     'LWout':      to_list(df_dash['LWout']),
#     'Pressure':   to_list(df_dash['Pressure']),
# }

# with open(DASHBOARD_DIR / 'data.json', 'w') as f:
#     json.dump(data, f, indent=4) # formatting json file for clarity

# print(f"[LOG] Dashboard data exported: {len(df_dash)} records to {DASHBOARD_DIR / 'data.json'}")
