
# Climate Change & Extreme Weather EDA

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load datasets
temp = pd.read_csv('temp.csv')
co2 = pd.read_csv('co2.csv')
weather = pd.read_csv('weather.csv')


# Clean Temperature data

temp_usa = temp[temp['Country'] == 'United States'].copy()
year_cols = [col for col in temp_usa.columns if str(col).strip().isdigit()]
temp_long = temp_usa.melt(
    id_vars=['Country'],
    value_vars=year_cols,
    var_name='Year',
    value_name='Temp_Anomaly_C'
)
temp_long['Year'] = temp_long['Year'].astype(int)
temp_long['Temp_Anomaly_C'] = pd.to_numeric(
    temp_long['Temp_Anomaly_C'], errors='coerce')


# Cleaning CO2 data

co2_world = co2[co2['Country'] == 'World'].copy()
co2_world['Year'] = co2_world['Date'].str[:4].astype(int)
co2_world = co2_world[['Year', 'Value']].rename(columns={'Value': 'CO2_PPM'})
co2_annual = co2_world.groupby('Year').mean().reset_index()


# Cleaning Extreme Weather data

weather.columns = weather.columns.str.strip()
weather['Begin_Date'] = pd.to_datetime(
    weather['Begin Date'], format='%Y%m%d', errors='coerce')
weather['Year'] = weather['Begin_Date'].dt.year
disasters_per_year = weather.groupby(
    'Year').size().reset_index(name='Num_Disasters')
disasters_per_year = disasters_per_year.dropna(subset=['Year'])
disasters_per_year['Year'] = disasters_per_year['Year'].astype(int)


# Merging datasets

climate_df = pd.merge(temp_long, co2_annual, on='Year', how='inner')
climate_df = pd.merge(climate_df, disasters_per_year, on='Year', how='left')
climate_df['Num_Disasters'] = climate_df['Num_Disasters'].fillna(0)

# Summary Statistics

print("\n--- Summary Statistics ---")
stats = climate_df[['Temp_Anomaly_C', 'CO2_PPM', 'Num_Disasters']].describe()
print(stats)

# Mode for each column
for col in ['Temp_Anomaly_C', 'CO2_PPM', 'Num_Disasters']:
    mode_val = climate_df[col].mode()[0]
    print(f"Mode of {col}: {mode_val}")


# Detecting Outliers

for col in ['Temp_Anomaly_C', 'CO2_PPM', 'Num_Disasters']:
    q1 = climate_df[col].quantile(0.25)
    q3 = climate_df[col].quantile(0.75)
    iqr = q3 - q1
    outliers = climate_df[(climate_df[col] < q1 - 1.5*iqr)
                          | (climate_df[col] > q3 + 1.5*iqr)]
    print(f"{col} outliers:\n", outliers[['Year', col]])


# Single variable plots with clean x-axis

fig, ax = plt.subplots(1, 3, figsize=(18, 4))

sns.lineplot(data=climate_df, x='Year',
             y='Temp_Anomaly_C', ax=ax[0], color='red')
ax[0].set_title('U.S. Temperature Anomalies')
ax[0].grid(True)
ax[0].set_xticks(range(climate_df['Year'].min(),
                 climate_df['Year'].max()+1, 5))
ax[0].set_xticklabels(range(climate_df['Year'].min(),
                      climate_df['Year'].max()+1, 5), rotation=45)

sns.lineplot(data=climate_df, x='Year', y='CO2_PPM', ax=ax[1], color='green')
ax[1].set_title('Global CO2 Concentration')
ax[1].grid(True)
ax[1].set_xticks(range(climate_df['Year'].min(),
                 climate_df['Year'].max()+1, 5))
ax[1].set_xticklabels(range(climate_df['Year'].min(),
                      climate_df['Year'].max()+1, 5), rotation=45)

# Extreme weather events as bar plot
sns.barplot(data=climate_df, x='Year',
            y='Num_Disasters', ax=ax[2], color='orange')
ax[2].set_title('Extreme Weather Events in USA')
ax[2].grid(True)
ax[2].set_xticks(range(0, len(climate_df['Year']), 5))
ax[2].set_xticklabels(climate_df['Year'][::5], rotation=45)

plt.tight_layout()
plt.show()


# Temperature vs CO2 (dual y-axis)

fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(climate_df['Year'], climate_df['Temp_Anomaly_C'],
         color='red', label='Temp Anomaly (°C)')
ax1.set_xlabel('Year')
ax1.set_ylabel('Temperature Anomaly (°C)', color='red')
ax1.tick_params(axis='y', labelcolor='red')
ax1.set_xticks(range(climate_df['Year'].min(), climate_df['Year'].max()+1, 5))
ax1.set_xticklabels(
    range(climate_df['Year'].min(), climate_df['Year'].max()+1, 5), rotation=45)

ax2 = ax1.twinx()
ax2.plot(climate_df['Year'], climate_df['CO2_PPM'],
         color='green', label='CO2 (ppm)')
ax2.set_ylabel('CO2 Concentration (ppm)', color='green')
ax2.tick_params(axis='y', labelcolor='green')

fig.suptitle('Temperature Anomalies vs Global CO2 Concentration', fontsize=14)
fig.tight_layout()
plt.show()


# Disasters vs Temp (dual y-axis)

fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.bar(climate_df['Year'], climate_df['Num_Disasters'],
        color='orange', alpha=0.6, label='Num Disasters')
ax1.set_xlabel('Year')
ax1.set_ylabel('Number of Extreme Weather Events', color='orange')
ax1.tick_params(axis='y', labelcolor='orange')
ax1.set_xticks(range(climate_df['Year'].min(), climate_df['Year'].max()+1, 5))
ax1.set_xticklabels(
    range(climate_df['Year'].min(), climate_df['Year'].max()+1, 5), rotation=45)

ax2 = ax1.twinx()
ax2.plot(climate_df['Year'], climate_df['Temp_Anomaly_C'],
         color='red', label='Temp Anomaly (°C)')
ax2.set_ylabel('Temperature Anomaly (°C)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

fig.suptitle(
    'Extreme Weather Events vs U.S. Temperature Anomalies', fontsize=14)
fig.tight_layout()
plt.show()


# Normalized overlay of all three variables

overlay_df = climate_df[['Year', 'Temp_Anomaly_C',
                         'CO2_PPM', 'Num_Disasters']].copy()
overlay_df['Temp_norm'] = (overlay_df['Temp_Anomaly_C'] - overlay_df['Temp_Anomaly_C'].min()) / \
    (overlay_df['Temp_Anomaly_C'].max() - overlay_df['Temp_Anomaly_C'].min())
overlay_df['CO2_norm'] = (overlay_df['CO2_PPM'] - overlay_df['CO2_PPM'].min()) / \
    (overlay_df['CO2_PPM'].max() - overlay_df['CO2_PPM'].min())
overlay_df['Disasters_norm'] = (overlay_df['Num_Disasters'] - overlay_df['Num_Disasters'].min()) / (
    overlay_df['Num_Disasters'].max() - overlay_df['Num_Disasters'].min())

plt.figure(figsize=(12, 5))
plt.plot(overlay_df['Year'], overlay_df['Temp_norm'],
         color='red', label='Temp Anomaly (normalized)')
plt.plot(overlay_df['Year'], overlay_df['CO2_norm'],
         color='green', label='CO2 (normalized)')
plt.plot(overlay_df['Year'], overlay_df['Disasters_norm'],
         color='orange', label='Extreme Weather (normalized)')
plt.xlabel('Year')
plt.ylabel('Normalized Values (0-1)')
plt.xticks(rotation=45)
plt.title('Overlay: Temperature, CO2, and Extreme Weather (Normalized)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# Time-series plots

plt.figure(figsize=(18, 4))

plt.subplot(1, 3, 1)
plt.plot(climate_df['Year'], climate_df['Temp_Anomaly_C'], color='red')
plt.xlabel('Year')
plt.ylabel('Temperature Anomaly (°C)')
plt.title('Temperature Anomaly Over Time')
plt.xticks(rotation=45)
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(climate_df['Year'], climate_df['CO2_PPM'], color='green')
plt.xlabel('Year')
plt.ylabel('CO2 Concentration (ppm)')
plt.title('CO2 Concentration Over Time')
plt.xticks(rotation=45)
plt.grid(True)

plt.subplot(1, 3, 3)
plt.bar(climate_df['Year'], climate_df['Num_Disasters'],
        color='orange', alpha=0.7)
plt.xlabel('Year')
plt.ylabel('Number of Extreme Weather Events')
plt.title('Extreme Weather Events Over Time')
plt.xticks(rotation=45)
plt.grid(True)

plt.tight_layout()
plt.show()


# Boxplots

plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
sns.boxplot(y=climate_df['Temp_Anomaly_C'])
plt.title('Temp Anomaly Boxplot')

plt.subplot(1, 3, 2)
sns.boxplot(y=climate_df['CO2_PPM'])
plt.title('CO2 Boxplot')

plt.subplot(1, 3, 3)
sns.boxplot(y=climate_df['Num_Disasters'])
plt.title('Extreme Weather Boxplot')
plt.tight_layout()
plt.show()


# Correlations

corr = climate_df[['Temp_Anomaly_C', 'CO2_PPM', 'Num_Disasters']].corr()
print("\n--- Correlation Matrix ---")
print(corr)

sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# Missing data check

missing = climate_df.isna().sum()
print("\n--- Missing Data ---")
print(missing)
