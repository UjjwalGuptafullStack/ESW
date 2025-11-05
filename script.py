import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load data
print("Loading data from feeds.csv...")
df = pd.read_csv('feeds.csv')

# Data preprocessing
print("Preprocessing data...")
df['created_at'] = pd.to_datetime(df['created_at'])
df['Timestamp'] = df['created_at']

# Rename fields for clarity
df['PM2.5'] = pd.to_numeric(df['field4'], errors='coerce')
df['PM10'] = pd.to_numeric(df['field5'], errors='coerce')
df['Temp_C'] = pd.to_numeric(df['field2'], errors='coerce')
df['Humidity_pct'] = pd.to_numeric(df['field3'], errors='coerce')

# Remove invalid readings (-1.00 values)
df.loc[df['PM2.5'] == -1.00, 'PM2.5'] = np.nan
df.loc[df['PM10'] == -1.00, 'PM10'] = np.nan
df.loc[df['Temp_C'] == -1.00, 'Temp_C'] = np.nan
df.loc[df['Humidity_pct'] == -1.00, 'Humidity_pct'] = np.nan

# Sort by timestamp
df = df.sort_values('Timestamp').reset_index(drop=True)

# Extract time features
df['hour'] = df['Timestamp'].dt.hour
df['date'] = df['Timestamp'].dt.date

print(f"Total records: {len(df)}")
print(f"Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
print(f"\nData summary:")
print(df[['PM2.5', 'PM10', 'Temp_C', 'Humidity_pct']].describe())

# Define Diwali date (you can modify this)
DIWALI_DATE = pd.to_datetime('2025-10-20', utc=True)  # Adjust to actual Diwali date in your data

# Create period categories
def categorize_period(timestamp):
    days_diff = (timestamp - DIWALI_DATE).days
    if days_diff < -1:
        return 'Pre-Diwali'
    elif -1 <= days_diff <= 1:
        return 'During Diwali'
    else:
        return 'Post-Diwali'

df['Period'] = df['Timestamp'].apply(categorize_period)

# AQI Calculation (India CPCB standards for PM2.5)
def calculate_aqi_pm25(pm25):
    if pd.isna(pm25):
        return np.nan
    if pm25 <= 30:
        return 'Good'
    elif pm25 <= 60:
        return 'Satisfactory'
    elif pm25 <= 90:
        return 'Moderate'
    elif pm25 <= 120:
        return 'Poor'
    elif pm25 <= 250:
        return 'Very Poor'
    else:
        return 'Severe'

df['AQI_Category'] = df['PM2.5'].apply(calculate_aqi_pm25)

# Create output directory for plots
import os
os.makedirs('analysis_plots', exist_ok=True)

print("\nGenerating analysis plots...\n")

# ========== ANALYSIS 1: Time-Series Trend Analysis ==========
print("1. Generating time-series trend analysis...")
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

axes[0].plot(df['Timestamp'], df['PM2.5'], color='red', alpha=0.7, linewidth=1)
axes[0].axvline(DIWALI_DATE, color='orange', linestyle='--', linewidth=2, label='Diwali')
axes[0].set_ylabel('PM2.5 (µg/m³)', fontsize=12)
axes[0].set_title('PM2.5 Concentration Over Time', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(df['Timestamp'], df['PM10'], color='blue', alpha=0.7, linewidth=1)
axes[1].axvline(DIWALI_DATE, color='orange', linestyle='--', linewidth=2, label='Diwali')
axes[1].set_xlabel('Time', fontsize=12)
axes[1].set_ylabel('PM10 (µg/m³)', fontsize=12)
axes[1].set_title('PM10 Concentration Over Time', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analysis_plots/1_timeseries_trend.png', dpi=300, bbox_inches='tight')
plt.close()

# ========== ANALYSIS 2: Temperature-PM Relationship ==========
print("2. Generating temperature-PM relationship analysis...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Filter valid data
temp_pm_data = df.dropna(subset=['Temp_C', 'PM2.5'])

if len(temp_pm_data) > 0:
    sns.regplot(x='Temp_C', y='PM2.5', data=temp_pm_data, ax=axes[0], 
                scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    axes[0].set_xlabel('Temperature (°C)', fontsize=12)
    axes[0].set_ylabel('PM2.5 (µg/m³)', fontsize=12)
    axes[0].set_title('Temperature vs PM2.5', fontsize=14, fontweight='bold')
    
    sns.regplot(x='Temp_C', y='PM10', data=temp_pm_data, ax=axes[1], 
                scatter_kws={'alpha':0.5}, line_kws={'color':'blue'})
    axes[1].set_xlabel('Temperature (°C)', fontsize=12)
    axes[1].set_ylabel('PM10 (µg/m³)', fontsize=12)
    axes[1].set_title('Temperature vs PM10', fontsize=14, fontweight='bold')
else:
    axes[0].text(0.5, 0.5, 'Insufficient temperature data', ha='center', va='center')
    axes[1].text(0.5, 0.5, 'Insufficient temperature data', ha='center', va='center')

plt.tight_layout()
plt.savefig('analysis_plots/2_temperature_pm_relationship.png', dpi=300, bbox_inches='tight')
plt.close()

# ========== ANALYSIS 3: Humidity Influence on PM ==========
print("3. Generating humidity-PM analysis...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

humidity_pm_data = df.dropna(subset=['Humidity_pct', 'PM2.5'])

if len(humidity_pm_data) > 0:
    sns.regplot(x='Humidity_pct', y='PM2.5', data=humidity_pm_data, ax=axes[0],
                scatter_kws={'alpha':0.5}, line_kws={'color':'green'})
    axes[0].set_xlabel('Humidity (%)', fontsize=12)
    axes[0].set_ylabel('PM2.5 (µg/m³)', fontsize=12)
    axes[0].set_title('Humidity vs PM2.5', fontsize=14, fontweight='bold')
    
    # Correlation heatmap
    corr_data = df[['PM2.5', 'PM10', 'Temp_C', 'Humidity_pct']].dropna()
    if len(corr_data) > 0:
        sns.heatmap(corr_data.corr(), annot=True, cmap='coolwarm', center=0,
                    ax=axes[1], fmt='.2f', square=True)
        axes[1].set_title('Correlation Matrix', fontsize=14, fontweight='bold')
else:
    axes[0].text(0.5, 0.5, 'Insufficient humidity data', ha='center', va='center')
    axes[1].text(0.5, 0.5, 'Insufficient data for correlation', ha='center', va='center')

plt.tight_layout()
plt.savefig('analysis_plots/3_humidity_pm_influence.png', dpi=300, bbox_inches='tight')
plt.close()

# ========== ANALYSIS 4: Diurnal Variation Profile ==========
print("4. Generating diurnal variation profile...")
hourly_avg = df.groupby('hour')[['PM2.5', 'PM10']].mean()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(hourly_avg.index, hourly_avg['PM2.5'], marker='o', color='red', 
        linewidth=2, markersize=8, label='PM2.5')
ax.plot(hourly_avg.index, hourly_avg['PM10'], marker='s', color='blue', 
        linewidth=2, markersize=8, label='PM10')
ax.set_xlabel('Hour of Day', fontsize=12)
ax.set_ylabel('Average Concentration (µg/m³)', fontsize=12)
ax.set_title('Diurnal Variation in PM Levels', fontsize=14, fontweight='bold')
ax.set_xticks(range(0, 24))
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('analysis_plots/4_diurnal_variation.png', dpi=300, bbox_inches='tight')
plt.close()

# ========== ANALYSIS 5: AQI Distribution ==========
print("5. Generating AQI category analysis...")
aqi_counts = df['AQI_Category'].value_counts()
aqi_order = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
aqi_counts = aqi_counts.reindex([cat for cat in aqi_order if cat in aqi_counts.index])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

colors = {'Good': 'green', 'Satisfactory': 'lightgreen', 'Moderate': 'yellow',
          'Poor': 'orange', 'Very Poor': 'red', 'Severe': 'darkred'}
bar_colors = [colors.get(cat, 'gray') for cat in aqi_counts.index]

aqi_counts.plot(kind='bar', ax=axes[0], color=bar_colors)
axes[0].set_xlabel('AQI Category', fontsize=12)
axes[0].set_ylabel('Number of Readings', fontsize=12)
axes[0].set_title('Distribution of Air Quality Categories', fontsize=14, fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)

# Pie chart
axes[1].pie(aqi_counts.values, labels=aqi_counts.index, autopct='%1.1f%%',
            colors=bar_colors, startangle=90)
axes[1].set_title('AQI Category Proportion', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('analysis_plots/5_aqi_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# ========== ANALYSIS 6: Temperature Inversion Indicator ==========
print("6. Generating temperature inversion analysis...")
temp_inversion_data = df.dropna(subset=['Temp_C', 'PM2.5'])

if len(temp_inversion_data) > 0:
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    color1 = 'tab:blue'
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Temperature (°C)', color=color1, fontsize=12)
    ax1.plot(temp_inversion_data['Timestamp'], temp_inversion_data['Temp_C'], 
             color=color1, alpha=0.7, linewidth=1.5)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('PM2.5 (µg/m³)', color=color2, fontsize=12)
    ax2.plot(temp_inversion_data['Timestamp'], temp_inversion_data['PM2.5'], 
             color=color2, alpha=0.7, linewidth=1.5)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title('Temperature Inversion Analysis (Temp vs PM2.5)', 
              fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig('analysis_plots/6_temperature_inversion.png', dpi=300, bbox_inches='tight')
    plt.close()
else:
    print("   Insufficient data for temperature inversion analysis")

# ========== ANALYSIS 7: Pre/During/Post Diwali Comparison ==========
print("7. Generating Diwali period comparison...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

period_data = df.dropna(subset=['PM2.5', 'PM10'])

if len(period_data) > 0:
    sns.boxplot(x='Period', y='PM2.5', data=period_data, ax=axes[0],
                order=['Pre-Diwali', 'During Diwali', 'Post-Diwali'],
                palette='Set2')
    axes[0].set_ylabel('PM2.5 (µg/m³)', fontsize=12)
    axes[0].set_xlabel('Period', fontsize=12)
    axes[0].set_title('PM2.5 Distribution by Period', fontsize=14, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=15)
    
    sns.boxplot(x='Period', y='PM10', data=period_data, ax=axes[1],
                order=['Pre-Diwali', 'During Diwali', 'Post-Diwali'],
                palette='Set2')
    axes[1].set_ylabel('PM10 (µg/m³)', fontsize=12)
    axes[1].set_xlabel('Period', fontsize=12)
    axes[1].set_title('PM10 Distribution by Period', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig('analysis_plots/7_diwali_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ========== ANALYSIS 8: Full Correlation Matrix ==========
print("8. Generating comprehensive correlation matrix...")
corr_cols = ['PM2.5', 'PM10', 'Temp_C', 'Humidity_pct']
corr_data = df[corr_cols].dropna()

if len(corr_data) > 0:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_data.corr(), annot=True, cmap='coolwarm', center=0,
                fmt='.3f', square=True, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title('Correlation Matrix of All Variables', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('analysis_plots/8_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

# ========== ANALYSIS 9: Cumulative Exposure (AOT) ==========
print("9. Generating cumulative exposure analysis...")
pm25_threshold = 60  # CPCB standard for PM2.5
exposure_data = df.dropna(subset=['PM2.5']).copy()

if len(exposure_data) > 0:
    exposure_data['Excess'] = (exposure_data['PM2.5'] - pm25_threshold).clip(lower=0)
    exposure_data['CumulativeExposure'] = exposure_data['Excess'].cumsum()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    axes[0].fill_between(exposure_data['Timestamp'], exposure_data['PM2.5'], 
                         pm25_threshold, where=(exposure_data['PM2.5'] > pm25_threshold),
                         color='red', alpha=0.3, label='Excess PM2.5')
    axes[0].plot(exposure_data['Timestamp'], exposure_data['PM2.5'], 
                color='darkred', linewidth=1)
    axes[0].axhline(pm25_threshold, color='orange', linestyle='--', 
                   linewidth=2, label=f'Threshold ({pm25_threshold} µg/m³)')
    axes[0].set_ylabel('PM2.5 (µg/m³)', fontsize=12)
    axes[0].set_title('PM2.5 Exposure Above Threshold', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(exposure_data['Timestamp'], exposure_data['CumulativeExposure'],
                color='purple', linewidth=2)
    axes[1].set_xlabel('Time', fontsize=12)
    axes[1].set_ylabel('Cumulative Excess PM2.5', fontsize=12)
    axes[1].set_title('Cumulative Exposure Over Time', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis_plots/9_cumulative_exposure.png', dpi=300, bbox_inches='tight')
    plt.close()

# ========== ANALYSIS 10: Vertical Dispersion Projection ==========
print("10. Generating vertical dispersion projection...")
# Use median PM2.5 during peak pollution for projection
peak_pm25 = df['PM2.5'].dropna().quantile(0.9)  # 90th percentile as "peak"
C0 = peak_pm25
k = 0.05  # Dispersion coefficient (empirical)

z_heights = np.arange(0, 100, 5)  # Heights from 0 to 100m
Cz = C0 * np.exp(-k * z_heights)

fig, ax = plt.subplots(figsize=(8, 10))
ax.plot(Cz, z_heights, linewidth=2, color='darkgreen', marker='o')
ax.axhline(12, color='red', linestyle='--', label='Node Height (~12m)')
ax.set_xlabel('Estimated PM2.5 Concentration (µg/m³)', fontsize=12)
ax.set_ylabel('Height Above Ground (m)', fontsize=12)
ax.set_title(f'Hypothetical Vertical PM2.5 Dispersion Profile\n(Peak Concentration: {C0:.1f} µg/m³)',
            fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('analysis_plots/10_vertical_dispersion.png', dpi=300, bbox_inches='tight')
plt.close()

# ========== SUMMARY STATISTICS ==========
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

print("\nOverall PM2.5 Statistics:")
print(df['PM2.5'].describe())

print("\nOverall PM10 Statistics:")
print(df['PM10'].describe())

print("\nPM Statistics by Period:")
period_stats = df.groupby('Period')[['PM2.5', 'PM10']].agg(['mean', 'median', 'max'])
print(period_stats)

print("\nAQI Category Distribution:")
print(df['AQI_Category'].value_counts())

# Save summary to text file
with open('analysis_plots/summary_statistics.txt', 'w') as f:
    f.write("="*60 + "\n")
    f.write("DIWALI AIR QUALITY ANALYSIS - SUMMARY STATISTICS\n")
    f.write("="*60 + "\n\n")
    f.write(f"Data Range: {df['Timestamp'].min()} to {df['Timestamp'].max()}\n")
    f.write(f"Total Records: {len(df)}\n")
    f.write(f"Valid PM2.5 Readings: {df['PM2.5'].notna().sum()}\n")
    f.write(f"Valid PM10 Readings: {df['PM10'].notna().sum()}\n\n")
    
    f.write("Overall PM2.5 Statistics:\n")
    f.write(df['PM2.5'].describe().to_string() + "\n\n")
    
    f.write("Overall PM10 Statistics:\n")
    f.write(df['PM10'].describe().to_string() + "\n\n")
    
    f.write("PM Statistics by Period:\n")
    f.write(period_stats.to_string() + "\n\n")
    
    f.write("AQI Category Distribution:\n")
    f.write(df['AQI_Category'].value_counts().to_string() + "\n")

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
print(f"\nAll plots saved in 'analysis_plots/' directory")
print("Summary statistics saved to 'analysis_plots/summary_statistics.txt'")
print("\nGenerated plots:")
print("  1. timeseries_trend.png - PM2.5 and PM10 over time")
print("  2. temperature_pm_relationship.png - Temperature vs PM correlation")
print("  3. humidity_pm_influence.png - Humidity effects and correlation matrix")
print("  4. diurnal_variation.png - Hourly pollution patterns")
print("  5. aqi_distribution.png - Air quality category distribution")
print("  6. temperature_inversion.png - Temperature-PM dual axis plot")
print("  7. diwali_comparison.png - Pre/During/Post Diwali boxplots")
print("  8. correlation_matrix.png - Full correlation heatmap")
print("  9. cumulative_exposure.png - Exposure above threshold")
print(" 10. vertical_dispersion.png - Hypothetical vertical profile")