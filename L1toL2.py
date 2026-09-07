import xarray as xr
from pathlib import Path
import os
import netCDF4 as nc
import json 
import pandas as pd
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from windrose import WindroseAxes
from scipy.signal import butter, filtfilt

DATA_FILE_1 = os.path.join("..", "data", "L1", "DataTable1.dat")
DATA_FILE_15 = os.path.join("..", "data", "L1", "DataTable15.dat")

# method for loading in data file
def load_toa5_table(path):
    names = pd.read_csv(path, nrows=0, header=0).columns
    df    = pd.read_csv(path, header=0, names=names, skiprows=0, na_values=['NAN'])
    df['LoggerTime'] = pd.to_datetime(df['LoggerTime'], errors='coerce')
    return df

# loading data and renaming columns
df15 = load_toa5_table(DATA_FILE_15)
df1  = load_toa5_table(DATA_FILE_1)

df15_subset = df15.sort_values('LoggerTime')[['LoggerTime','Temp1','Temp2','TiltX','TiltY', 'Hum1', 'Hum2']]
df1 = pd.merge_asof(df1.sort_values('LoggerTime').reset_index(drop=True), df15_subset, on='LoggerTime', direction='backward')
# Fill rows before df15 starts (backward match finds nothing) and any NaN gaps in df15
for col in ['Temp1', 'Temp2', 'TiltX', 'TiltY', 'SWin', 'SWout', 'LWin', 'LWout', 'Hum1', 'Hum2']:
    df1[col] = df1[col].bfill().ffill()


# SMOOTHING AND AVERAGING STATION HEIGHT
# linear interpolating nan values so smoothing can be applied

threedaywindow = 288
df15['Altitude'] = df15['Altitude'].interpolate().ffill().bfill()
days = (df15['LoggerTime'] - df15['LoggerTime'].iloc[0]).dt.total_seconds()/86400
slope2, slope, intercept = np.polyfit(days[df15["Altitude"].notna()], df15.loc[df15["Altitude"].notna(), "Altitude"], 2)
df15['AltitudeTrend'] = slope2 * days**2 + slope * days + intercept
total_drop = df15['AltitudeTrend'].iloc[-1] - df15['AltitudeTrend'].iloc[0]
print(f"Altitude trend: {slope*100:.4f} cm/day, total drop over {days.iloc[-1]:.2f} dager: {total_drop:.2f} m")
Altitude_melt_trend = f"{slope*1000:.2f} mm/dag" # in cm/day
Altitude_total_drop = f"{total_drop:.2f} m" # in m
# SMOOTHING PRESSURE - don't want this, only for dashboard
df15['Pressure'] = df15['Pressure'].rolling(window=1, center=True, min_periods=1).mean()

# CORRECTING SOLAR RADIATION BASED ON ZENITH ANGLE AND CLOUD COVER

# CLOUD COVER
# This is clearly not the best way of doing this, as the results are not so good
TempAtmos = df1['Temp2']
Eps = 5.670374419e-8
T0 = 273.15
# Theoretical lower and upper bounds for atmospheric longwave radiation
LW_overcast = Eps*(TempAtmos+T0)**4 # assumption
LW_clear = (9.36508e-6)*Eps*(TempAtmos+T0)**6 # Swinbank(1963) approximation for BB rad from clear sky
CloudCover = (df1['LWin']-LW_clear)/(LW_overcast-LW_clear) # assuming cloud cover fraction is linear with LW radiation (Harrison et al 2008)
CloudCoverScaled = CloudCover.clip(lower=0,upper=1)#(CloudCover - CloudCover.min()) / (CloudCover.max() - CloudCover.min())
DiffuseFraction = 0.2+0.8*CloudCoverScaled # following Fasuto 2021 method
(df1['CloudCoverScaled'], df1['DiffuseFraction']) = (CloudCoverScaled, DiffuseFraction) #saving in data frame

# ZENITH ANGLE
DayYear = df1["LoggerTime"].dt.dayofyear
Hour = df1["LoggerTime"].dt.hour
Minute = df1["LoggerTime"].dt.minute
lon = df15['Longitude'][1]
lat = df15['Latitude'][1]
# Day angle in radians
DayAngle = 2. * np.pi * (DayYear + (Hour + Minute / 60.) / 24. - 1.) / 365.
# Solar declination from fundamentals
Declination = np.arcsin((0.006918 - 
                     0.399912 * np.cos(DayAngle) + 0.070257 * np.sin(DayAngle) - 
                     0.006758 * np.cos(2 * DayAngle) + 0.000907 * np.sin(2 * DayAngle) - 
                     0.002697 * np.cos(3 * DayAngle) + 0.00148 * np.sin(3 * DayAngle)))
HourAngle = 2. * np.pi * (((Hour + Minute / 60.) / 24. - 0.5) + lon / 360.) #positive when east
DirectionSun_deg = HourAngle * 180. / np.pi - 180.
DirectionSun_deg = np.where(DirectionSun_deg < 0, DirectionSun_deg + 360., DirectionSun_deg)
lat_rad = np.radians(lat)
zenith_cos = (np.cos(lat_rad) * np.cos(Declination) * np.cos(HourAngle) + 
              np.sin(lat_rad) * np.sin(Declination))
ZenithAngle_rad = np.arccos(np.clip(zenith_cos, -1.0, 1.0)) # clipped to avoid floating point errors
df1['ZenithAngle_deg'] = np.degrees(ZenithAngle_rad)

# THEORETICAL SOLAR RADIATION
SRtoa = 1361. * np.cos(ZenithAngle_rad)
SRtoa = np.where(df1['ZenithAngle_deg'] >= 90., 0., SRtoa)
ZenithAngle_rad_above_horizon = np.clip(ZenithAngle_rad, 0, np.pi/2) # Esimating the transmittance of the atmosphere
tb = 0.268 + 0.645*np.exp(-0.285/np.cos(ZenithAngle_rad_above_horizon)) # beam transmissivity (Hottel 1976)
td = 0.271 - 0.294 * tb #diffuse transmissivity (Hottel 1976)
df1['IncomingTheoretical_clear'] = SRtoa*tb + SRtoa*td # Estimating the clear-sky incoming SW radiation
df1['IncomingTheoretical_with_clouds'] = SRtoa*tb*(1-CloudCoverScaled) + SRtoa*td + SRtoa*tb*CloudCoverScaled*DiffuseFraction # Cloud estimate not good

# Time of highest radiation on very clear days - need to compare to theoretical solar times
# This is to calculate how much the station has rotated at each point
#SWdf = df1[["LoggerTime", "SWin"]]
#SWdf['SWin'] = uniform_filter1d(df1['SWin'].values.flatten(), size=threedaywindow, mode="reflect")
#SWdf = SWdf[SWdf["SWin"] > 790].reset_index(drop=True)
#SWmax_index = SWdf.groupby(SWdf["LoggerTime"].dt.date)["SWin"].idxmax()
#SWmax = SWdf.loc[SWmax_index].reset_index(drop=True)

# CNR4 TILT CORRECTION 
# Interpolating nans to prep for filtering, using intial tilt as baseline, 1 hour smoothing with mirroring
(tx0, ty0) = (-0.697, 1.374)
hourwindow = 60;
(TiltX, TiltY) = (df1["TiltX"] - tx0, df1["TiltY"] - ty0)
TiltX_smoothed = uniform_filter1d(TiltX.values.flatten(), size=hourwindow, mode="reflect")
TiltY_smoothed = uniform_filter1d(TiltY.values.flatten(), size=hourwindow, mode="reflect")
(df1['TiltX'], df1['TiltY']) = (TiltX_smoothed, TiltY_smoothed)

# TILT CORRECTION
# measured +X=+South=Pitch, +Y=-West=-Roll based on measurement standards. Using smoothed data.
(roll, pitch) = (-np.deg2rad(TiltY_smoothed), np.deg2rad(TiltX_smoothed))
# Cartesian converstion: X axis E-W, Y axis N-S, Z axis vertical.
X = np.sin(roll) * np.cos(roll) * (np.sin(pitch)**2) + np.sin(roll) * (np.cos(pitch)**2)
Y = np.sin(pitch) * np.cos(pitch) * (np.sin(roll)**2) + np.sin(pitch) * (np.cos(roll)**2)
Z = np.cos(roll) * np.cos(pitch) + (np.sin(roll)**2) * (np.sin(pitch)**2)
# spherical coordinate conversion
SensorPhi = (-np.pi/2.0 - np.arctan2(Y, X))
SensorTheta = (np.arccos(Z / (X**2 + Y**2 + Z**2)**0.5))
#Correction uses diffuse fraction, which is not a good approximation. 
CorrectionFactor = (np.sin(Declination) * np.sin(lat_rad) * np.cos(SensorTheta) 
- np.sin(Declination) * np.cos(lat_rad) * np.sin(SensorTheta) * np.cos(SensorPhi + np.pi) 
+ np.cos(Declination) * np.cos(lat_rad) * np.cos(SensorTheta)  * np.cos(HourAngle) 
+ np.cos(Declination) * np.sin(lat_rad) * np.sin(SensorTheta) * np.cos(SensorPhi + np.pi) * np.cos(HourAngle) 
+ np.cos(Declination) * np.sin(SensorTheta) * np.sin(SensorPhi + np.pi) * np.sin(HourAngle));
CorrectionFactor = np.cos(ZenithAngle_rad)/CorrectionFactor
CorrectionFactor = np.where((CorrectionFactor < 0) | (df1['ZenithAngle_deg'] > 90), 1.0, CorrectionFactor) #if sun out of view of sensor, no correction
#only direct sunlight is tilted: Only (1-DiffuseFraction) is direct and is affected by tilt
df1['TiltCorrectionFactor'] = CorrectionFactor / (1 - df1['DiffuseFraction'] + CorrectionFactor * df1['DiffuseFraction'])
df1['SWin'] = uniform_filter1d(df1['SWin'].values.flatten(), size=hourwindow, mode="reflect")
df1['SWinTiltCorrected'] = df1['SWin'] * df1['TiltCorrectionFactor']

# ALBEDO 
#SWin_range = uniform_filter1d(df1['SWinTiltCorrected'].values.flatten(), size=hourwindow*24, mode="reflect")
df1['SWin_range'] = df1['SWinTiltCorrected'].where((df1['SWinTiltCorrected'] > 20) & (df1['ZenithAngle_deg'] < 60))
SWout_range = df1['SWout'].where(df1['SWout'] > 0)
df1['Albedo'] = (SWout_range/df1['SWin_range'])
df1['Albedo'] = df1['Albedo'].where((df1['Albedo'] < 1) & (df1['Albedo'] > 0))
df1['Albedo'] = df1['Albedo'].rolling(window=hourwindow, center=True, min_periods=1).mean()

# CLEARNESS INDEX
# Calcluating albedo at midday on clear days and classifying.
df1['ClearnessIndex'] = 1 - df1['CloudCoverScaled'].rolling(window=hourwindow, center=True, min_periods=1).mean().where((df1['SWinTiltCorrected'] > 20) & (df1['ZenithAngle_deg'] < 60)) #df1['SWin_range']/SRtoa #
df1['ClearnessIndexMean'] = df1['ClearnessIndex'].groupby(df1["LoggerTime"].dt.date).transform('mean')

# SURFACE CONDITIONS
df1['ClearMiddayAlbedo'] = df1['Albedo'].where((df1['ClearnessIndex'] >= 0.6) & (df1["LoggerTime"].dt.time >= pd.Timestamp("11:44").time())
     & (df1["LoggerTime"].dt.time <= pd.Timestamp("11:45").time()))
df1['MeanClearMiddayAlbedo'] = df1['ClearMiddayAlbedo'].groupby(df1["LoggerTime"].dt.date).transform('mean')
print(f"Mean clear midday albedo: {df1['MeanClearMiddayAlbedo'].mean():.3f} (std: {df1['MeanClearMiddayAlbedo'].std():.3f})")
bins = [0, 0.34, 0.43, 0.51, 0.66, 0.69, 0.80, 0.88, 0.97]
labels = ['Contaminated', 'Bar is', 'Bar is/firn', 'Firn', 'Firn/smeltende snø', 'Smeltende snø', 'Kompakt snø', 'Nysnø']
df1['SurfaceCondition'] = pd.cut(df1['MeanClearMiddayAlbedo'], bins=bins, labels=labels, right=False)
df1['SurfaceCondition'] = df1['SurfaceCondition'].ffill().bfill() #filling in with last availbale snow condition

# SURFACE TEMPERATURE
emissivity = 0.98 #assuming fresh snow for now
df1['TempSurface'] = ((df1['LWout']-(1-emissivity)*df1['LWin'])/emissivity/5.67e-8)**0.25 - T0
df1['TempSurface'] = df1['TempSurface'].rolling(window=hourwindow, center=True, min_periods=1).mean()
df1['TempSurface'] = df1['TempSurface'].clip(upper=0) #cannot be above zero

# TIME SINCE LAST RAIN
TimeSinceLastRain = df1['LoggerTime'].iloc[-1] - df1['LoggerTime'].where(df1['Precip'] > 0).ffill().iloc[-1]
total_s = int(TimeSinceLastRain.total_seconds())
_d, rem = divmod(total_s, 86400)
_h, rem = divmod(rem, 3600)
_m = rem // 60
if _d > 0:
    TimeSinceLastRainNO = f"{_d} {'dag' if _d == 1 else 'dager'} {_h} {'time' if _h == 1 else 'timer'} siden"
elif _h > 0:
    TimeSinceLastRainNO = f"{_h} {'time' if _h == 1 else 'timer'} {_m} min siden"
else:
    TimeSinceLastRainNO = f"{_m} min siden"

#WIND
# fill WindMax values with the maximum value in each 15-minute period
df1['WindMax'] = df1.groupby(pd.Grouper(key='LoggerTime', freq='30min'))['WindMax'].transform('max').rolling(window=int(hourwindow/2), center=True, min_periods=1).mean()
df1['WindSpeed'] = df1['WindSpeed']#.rolling(window=int(hourwindow), center=True, min_periods=1).mean()
df1['WindDir'] = df1['WindDir'].rolling(window=int(hourwindow/2), center=True, min_periods=1).mean()
df1['WindDirAvg'] = df1['WindDir'].rolling(window=int(hourwindow), center=True, min_periods=1).mean()


print(f"Current surface condition:\n{df1['SurfaceCondition'].iloc[-1]}")
print(f"Time of last measurement: {df1['LoggerTime'].iloc[-1]}")
print(f"Time since last rain: {TimeSinceLastRain}")
print(f"Current surface temperature: {df1['TempSurface'].iloc[-1]:.2f} °C")
print(f"Sky is {100-df1['ClearnessIndex'].iloc[-1] *100:.1f} % overcast")
print(f"Current wind speed: {df1['WindSpeed'].iloc[-1]:.2f} m/s, direction: {df1['WindDir'].iloc[-1]:.1f}°")
# however, we know that emissivity changes with snow conditions, will need to fix with albedo.
#TempSurface = ((df1['LWout']-(1-emissivity)*df1['LWin'])/emissivity/5.67e-8)**0.25 - T0
#TempSurface = TempSurface.clip(upper=0) #cannot be above zero



# CHANGE IN SURFACE TEMPERATURE



# ax = WindroseAxes.from_ax()
# ax.bar(
#     df1["WindDir"],
#     df1["WindSpeed"],
#     normed=True,
#     opening=0.8,  # Width of the bars (0.8 = 80% width)
#     edgecolor="white",  # Thin border line between segments
#     cmap=plt.cm.viridis,  # High-contrast color palette
# )
# ax.set_legend(title="Wind Speed (m/s)", loc="lower left", bbox_to_anchor=(1.1, 0))
# plt.title("Wind Rose")
# plt.show()

last = 500
#last=8354
#last =129565
#x=np.arange(last - 1)
#plt.figure(figsize=(15, 5)); 
#plt.plot(x,CloudCoverScaled[-last:-1], label="Theoretical (clear sky)", alpha=0.7); 
#plt.plot(x,DiffuseFraction[-last:-1], label="Incoming SW (smoothed)", alpha=0.7); 
#plt.plot(x,IncomingTheoretical_with_clouds[-last:-1], label="Theoretical (with clouds)", alpha=0.7); 
#plt.plot(x,df1['SWinTiltCorrected'][-last:-1], label="SW in tilt corrected", alpha=0.7); 
#plt.plot(df1['LoggerTime'][-last:-1],df1['WindDirAvg'][-last:-1], label="Wind direction", alpha=0.8); 
#plt.plot(df1['LoggerTime'][-last:-1], df1['WindSpeed'].rolling(window=hourwindow, center=True, min_periods=1).mean()[-last:-1]*10, label="Wind Speed (smoothed)", alpha=0.4);
#plt.plot(df1['LoggerTime'][-last:-1], df1['WindMax'][-last:-1]*10, label="Wind gusts (scaled)", alpha=0.4);
#df1['CloudCoverScaled'] = df1['CloudCoverScaled'].rolling(window=hourwindow, center=True, min_periods=1).mean().where((df1['SWinTiltCorrected'] > 20) & (df1['ZenithAngle_deg'] < 60))
#plt.plot(x, df15['Pressure'][-last:-1], label="Altitude data (noisy)", alpha=0.3); 

#plt.plot(df1['LoggerTime'][-last:-1], (1-df1['CloudCoverScaled'])[-last:-1], label="Clearness based on LW", alpha=0.7); 
#plt.plot(df1['LoggerTime'][-last:-1], df1['ClearnessIndex'].groupby(df1["LoggerTime"].dt.date).transform('mean')[-last:-1], label="Clearness based on theoretical vs observed incoming SW (daily means)", alpha=0.7); 
#plt.plot(df1['LoggerTime'][-last:-1], (1-df1['CloudCoverScaled'].groupby(df1["LoggerTime"].dt.date).transform('mean'))[-last:-1], label="Clearness based on LW (daily means)", alpha=0.7); 


#plt.plot(x,df1['SWin'][-last:-1], label="SW in", alpha=0.7); 
#plt.plot(x,df1['IncomingTheoretical_clear'][-last:-1], label="Theoretical (clear)", alpha=0.7); 
#plt.plot(x,df1['TiltX'][-last:-1], label="Theoretical (clear)", alpha=0.7); 


#plt.axhline(y=0.51, color='gray', linestyle='--', alpha=0.5); plt.axhline(y=0.66, color='gray', linestyle='--', alpha=0.5); plt.axhline(y=0.69, color='gray', linestyle='--', alpha=0.5); plt.axhline(y=0.80, color='gray', linestyle='--', alpha=0.5); plt.axhline(y=0.88, color='gray', linestyle='--', alpha=0.5);
#plt.axhline(y=0.8);plt.axhline(y=0.6);plt.axhline(y=0.4)
#plt.plot(x,LW_clear[-last:-1], label="LW clear (based on Swinback 1963 assumption)", alpha=0.7); 
#plt.plot(x,Altitude_smoothed[-last:-1], label="smoothed"); 
#plt.legend(loc='lower left');
#plt.title(f"Altitude trend"); 
#plt.show()

#RADIAL WIND DIRECTION SPIRAL
# spiral_df = df1[['LoggerTime', 'WindDir', 'WindSpeed']].copy().dropna(subset=['WindDir'])
# spiral_df = spiral_df[spiral_df['LoggerTime'] >= spiral_df['LoggerTime'].max() - pd.Timedelta(days=20)]
# spiral_df['WindDir_rad'] = np.deg2rad(spiral_df['WindDir'])
# spiral_df['ElapsedHours'] = (spiral_df['LoggerTime'] - spiral_df['LoggerTime'].iloc[0]).dt.total_seconds() / 3600.0
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection='polar')
# ax.plot(spiral_df['WindDir_rad'], spiral_df['ElapsedHours'], color='tab:blue', linewidth=0.8, alpha=0.7)
# sc = ax.scatter(spiral_df['WindDir_rad'], spiral_df['ElapsedHours'], c=spiral_df['WindSpeed'], cmap='viridis', s=12, alpha=0.8)
# ax.set_theta_zero_location('N')
# ax.set_theta_direction(-1)
# ax.set_rlabel_position(180)
# ax.set_title('Wind direction evolution spiral')
# cbar = plt.colorbar(sc, ax=ax, pad=0.1)
# cbar.set_label('Wind speed (m/s)')
# plt.show()

#DROPPING UNNEEDED COLUMNS
df1 = df1.drop(columns=['F2_Max'], errors='ignore') # not needed for further analysis
df15 = df15.drop(columns=['Temp1', 'Temp2', 'TiltX', 'TiltY', 'Hum1', 'Hum2'], errors='ignore')

print(f"df1 columns: {df1.columns.tolist()}")
print(f"df15 columns: {df15.columns.tolist()}")

# #SAVING CLEANED DATASET
#Save processed L2 data for further analysis.
L2_DATA_1 = os.path.join("..", "data", "L2", "DataTable1.dat")
L2_DATA_15 = os.path.join("..", "data", "L2", "DataTable15.dat")
df1.to_csv(L2_DATA_1, index=False, sep=',')
df15.to_csv(L2_DATA_15, index=False, sep=',')
print(f"[LOG] Saved L2 data to: {L2_DATA_1} and  {L2_DATA_15}")


##THIS IS AFTER ALL CALUCATIONS AND CORRECTIONS
## ALL THIS BELOW NEEDS TO GO INTO L2TODASHBOARD BUT THIS
## ALSO MEANS CHANGING THE CRON JOBS AND INDEX.HTML DEPENDENCY.
#RESAMPLING AND MERGING DATA
# Resample 1-min data to 15-min intervals
# Actually don't want to resample, but just merge and dropna for plotting.
# In the end I want to save in netcdf with metadata anyway.
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


# #DASHBOARD EXPORT
# filter to the last 7 days, and write data.json for the website.

DASHBOARD_DIR = Path(__file__).parent.parent / 'dashboard'

# weekdata1 = df1[df1['LoggerTime'] >= df1['LoggerTime'].max() - pd.Timedelta(days=7)]
# weekdata15 = df15[df15['LoggerTime'] >= df15['LoggerTime'].max() - pd.Timedelta(days=7)]
weekdata1 = df1[df1['LoggerTime'] >= df1['LoggerTime'].max() - pd.Timedelta(3, unit='D')]
weekdata15 = df15[df15['LoggerTime'] >= df15['LoggerTime'].max() - pd.Timedelta(3, unit='D')]

weekdata15['Pressure'] = weekdata15['Pressure'].rolling(window=60, center=True, min_periods=1).mean()
weekdata15 = weekdata15[['LoggerTime', 'Altitude', 'AltitudeTrend', 'Pressure']].sort_values('LoggerTime')
weekdata1 = weekdata1[['LoggerTime', 'Temp1', 'Hum2', 'WindSpeed', 'WindDir', 'WindMax', 'SWin', 'SWout', 'LWin', 'LWout', 'ClearnessIndex', 'SurfaceCondition', 'Albedo']].sort_values('LoggerTime')
#smooth temp and humidity for dashboard
weekdata1['Temp1'] = weekdata1['Temp1'].rolling(window=60*12, center=True, min_periods=1).mean()
weekdata1['Hum2'] = weekdata1['Hum2'].rolling(window=60*12, center=True, min_periods=1).mean()

df_dash = pd.merge_asof(
     weekdata1.sort_values('LoggerTime'),
     weekdata15,
     on='LoggerTime', direction='nearest', tolerance=pd.Timedelta('8min')
)

def to_list(series):
    return [None if pd.isna(v) else round(float(v), 2) for v in series]

data = {
    'updated':    pd.Timestamp.now().isoformat(),
    't':          [ts.isoformat() for ts in df_dash['LoggerTime']],
    'Temp1':      to_list(df_dash['Temp1']),
    'Hum1':       to_list(df_dash['Hum2']),
    'WindSpeed':  to_list(df_dash['WindSpeed']),
    'WindDir':    to_list(df_dash['WindDir']),
    'WindMax':    to_list(df_dash['WindMax']),
    'SWin':       to_list(df_dash['SWin']),
    'SWout':      to_list(df_dash['SWout']),
    'LWin':       to_list(df_dash['LWin']),
    'LWout':      to_list(df_dash['LWout']),
    'Pressure':   to_list(df_dash['Pressure']),
    'Altitude':   to_list(df_dash['Altitude']),
    'AltitudeTrend': to_list(df_dash['AltitudeTrend']),
    'ClearnessIndex': to_list(df_dash['ClearnessIndex']),
    'SurfaceCondition': [str(df_dash['SurfaceCondition'].iloc[-1])],
    'Albedo': to_list(df_dash['Albedo']),
    'TimeSinceLastRain': TimeSinceLastRainNO,
    'Altitude_melt_trend': str(Altitude_melt_trend),
    'Altitude_total_drop': str(Altitude_total_drop)
}

with open(DASHBOARD_DIR / 'data.json', 'w') as f:
     json.dump(data, f, indent=4) # formatting json file for clarity

print(f"[LOG] Dashboard data exported: {len(df_dash)} records to {DASHBOARD_DIR / 'data.json'}")

# WINDROSE — past 30 days from last measurement
wind7      = df1[df1['LoggerTime'] >= df1['LoggerTime'].max() - pd.Timedelta(30, unit='D')]
all_wind   = wind7['WindSpeed'].dropna()
wind_dir   = wind7[['WindDir', 'WindSpeed']].dropna(subset=['WindDir'])
calm_pct   = (all_wind <= 3).mean() * 100
total      = len(all_wind)

n_sectors     = 16
sector_size   = 360.0 / n_sectors
sector_centers= np.arange(n_sectors) * sector_size
angles_rad    = np.deg2rad(sector_centers)
compass       = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW']
speed_bins    = [3,  5,  8, 11, 15, np.inf]
speed_labels  = ['3–5','5–8','8–11','11–15','>15']
wr_colors     = ["#94b594",'#78a057','#415240','#1d401d', "#0C200E"]

counts = np.zeros((len(speed_labels), n_sectors))
for j, center in enumerate(sector_centers):
    lo = (center - sector_size / 2) % 360
    hi = (center + sector_size / 2) % 360
    dm = (wind_dir['WindDir'] >= lo) & (wind_dir['WindDir'] < hi) if lo < hi \
         else (wind_dir['WindDir'] >= lo) | (wind_dir['WindDir'] < hi)
    for i, (sl, sh) in enumerate(zip(speed_bins[:-1], speed_bins[1:])):
        counts[i, j] = (dm & (wind_dir['WindSpeed'] >= sl) & (wind_dir['WindSpeed'] < sh)).sum()

freq    = counts / total * 100 if total > 0 else counts
bottoms = np.zeros(n_sectors)
bar_w   = 2 * np.pi / n_sectors * 0.85

BG, FG, MID = '#e8f4e0', '#1d401d', '#78a057'
fig = plt.figure(figsize=(7, 7), facecolor=BG)
ax  = fig.add_subplot(111, projection='polar')
ax.set_facecolor(BG)
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
for color, label, row in zip(wr_colors, speed_labels, freq):
    ax.bar(angles_rad, row, width=bar_w, bottom=bottoms,
           color=color, edgecolor=BG, linewidth=0.4, label=f'{label} m/s')
    bottoms += row

ax.set_xticks(angles_rad)
ax.set_xticklabels(compass, fontsize=10, color=FG)
ax.set_rlabel_position(135)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
ax.tick_params(axis='y', labelsize=8, colors=MID)
ax.set_ylim(0, bottoms.max() * 1.2)
ax.grid(True, linestyle='--', alpha=0.3, color=MID)
ax.spines['polar'].set_visible(False)

fig.suptitle(f'Vindrose - siste 30 dager',
             fontsize=15, color=FG, y=0.98)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in wr_colors]
leg = fig.legend(handles, [f'{l} m/s' for l in speed_labels],
           loc='lower center', bbox_to_anchor=(0.5, 0.01),
           ncol=5, fontsize=11, edgecolor='#9db89d', facecolor=BG, labelcolor=FG)
leg.get_title().set_color(MID)

plt.tight_layout(rect=[0, 0.07, 1, 0.95])
plt.savefig(DASHBOARD_DIR / 'windrose.png', dpi=150, facecolor=BG)
plt.close()
print(f"[LOG] Windrose saved to {DASHBOARD_DIR / 'windrose.png'}")

# TEMPERATURE, Cloud Cover AND PRESSURE PLOTS — past 30 days, same dimensions as windrose
CREAM      = '#e8f4e0'
DK_GREEN   = '#1d401d'
SOFT_RED   = "#cd625e"
MUTED_BLUE = "#5396c9"
CMT_GREY   = '#7f8c8d'

plot_2w1  = df1[ df1['LoggerTime']  >= df1['LoggerTime'].max()  - pd.Timedelta(30, unit='D')]
plot_2w15 = df15[df15['LoggerTime'] >= df15['LoggerTime'].max() - pd.Timedelta(30, unit='D')]
temp_smooth = plot_2w1['Temp1'].rolling(window=120, center=True, min_periods=1).mean()
cc_smooth  = plot_2w1['CloudCoverScaled'].rolling(window=120, center=True, min_periods=1).mean()*100
pres_smooth = plot_2w15['Pressure'].rolling(window=16, center=True, min_periods=1).mean()

fig2, (ax_t, ax_h, ax_p) = plt.subplots(3, 1, figsize=(7, 7), facecolor=CREAM, sharex=True)
fig2.patch.set_facecolor(CREAM)
for ax in (ax_t, ax_h, ax_p):
    ax.set_facecolor(CREAM)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(DK_GREEN)
    ax.spines['left'].set_color(DK_GREEN)
    ax.tick_params(colors=DK_GREEN, labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.35, color=DK_GREEN)

ax_t.plot(plot_2w1['LoggerTime'], temp_smooth, color=SOFT_RED, linewidth=1.5)
ax_t.set_ylabel('Temp (°C)', color=DK_GREEN, fontsize=15)

ax_h.plot(plot_2w1['LoggerTime'], cc_smooth, color=MUTED_BLUE, linewidth=1.5)
ax_h.set_ylabel('Skydekke (%)', color=DK_GREEN, fontsize=15)
ax_h.set_ylim(0, 100)

ax_p.plot(plot_2w15['LoggerTime'], pres_smooth, color=CMT_GREY, linewidth=1.5)
ax_p.set_ylabel('Trykk (hPa)', color=DK_GREEN, fontsize=15)
ax_p.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
ax_p.xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.setp(ax_p.xaxis.get_majorticklabels(), rotation=30, ha='right', color=DK_GREEN)

fig2.suptitle('Temperatur, skydekke og trykk – siste 30 dager',
              color=DK_GREEN, fontsize=15, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.96], pad=0.8, h_pad=0.4)
plt.savefig(DASHBOARD_DIR / 'temp_pressure_plot.png', dpi=150, facecolor=CREAM)
plt.close()
print(f"[LOG] Temperature/humidity/pressure plot saved to {DASHBOARD_DIR / 'temp_pressure_plot.png'}")










