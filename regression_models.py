
# Climate Change & Extreme Weather: Robust OLS Regression with VIF
# import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tools import add_constant
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro


# Load & clean datasets

temp = pd.read_csv('temp.csv')
co2 = pd.read_csv('co2.csv')
weather = pd.read_csv('weather.csv')

# Temperature (USA)
temp_usa = temp[temp['Country'] == 'United States'].copy()
year_cols = [col for col in temp_usa.columns if col.isdigit()]
temp_long = temp_usa.melt(
    id_vars=['Country', 'ISO2', 'ISO3', 'Indicator', 'Unit', 'Source', 'CTS Code',
             'CTS Name', 'CTS Full Descriptor'],
    value_vars=year_cols,
    var_name='Year',
    value_name='Temp_Anomaly_C'
)
temp_long['Year'] = temp_long['Year'].astype(int)
temp_long['Temp_Anomaly_C'] = pd.to_numeric(
    temp_long['Temp_Anomaly_C'], errors='coerce')

# CO2 (World)
co2_world = co2[co2['Country'] == 'World'].copy()
co2_world['Year'] = co2_world['Date'].str[:4].astype(int)
co2_world = co2_world[['Year', 'Value']].rename(columns={'Value': 'CO2_PPM'})
co2_annual = co2_world.groupby('Year').mean().reset_index()

# Extreme Weather (USA)
weather.columns = weather.columns.str.strip()
weather['Begin_Date'] = pd.to_datetime(
    weather['Begin Date'], format='%Y%m%d', errors='coerce')
weather['Year'] = weather['Begin_Date'].dt.year
disasters_per_year = weather.groupby(
    'Year').size().reset_index(name='Num_Disasters')
disasters_per_year['Year'] = disasters_per_year['Year'].astype(int)

# Merge datasets
climate_df = pd.merge(temp_long, co2_annual, on='Year', how='inner')
climate_df = pd.merge(climate_df, disasters_per_year, on='Year', how='left')
climate_df['Num_Disasters'] = climate_df['Num_Disasters'].fillna(0).astype(int)

# Working dataframe
df_reg = climate_df[['Temp_Anomaly_C', 'CO2_PPM', 'Num_Disasters']].dropna()


# Prepare data for regression

X = add_constant(df_reg[['Temp_Anomaly_C', 'CO2_PPM']])
y = df_reg['Num_Disasters']

# Breusch-Pagan test for heteroscedasticity

ols_test = sm.OLS(y, X).fit()
bp_test = het_breuschpagan(ols_test.resid, X)
bp_labels = ["LM Stat", "LM p-value", "F Stat", "F p-value"]
print("\nBreusch-Pagan Test (OLS Residuals):")
print(dict(zip(bp_labels, bp_test)))


# Fit robust OLS (HC3) regression

robust_ols = sm.OLS(y, X).fit(cov_type='HC3')


# Standardized residuals & Shapiro-Wilk normality

std_resid = robust_ols.resid / np.sqrt(robust_ols.scale)

plt.scatter(robust_ols.fittedvalues, std_resid)
plt.axhline(0, linestyle='--')
plt.xlabel('Fitted values')
plt.ylabel('Standardized Residuals')
plt.title('Standardized Residuals vs Fitted (Robust OLS)')
plt.show()

shapiro_test = shapiro(std_resid)
print("\nShapiro-Wilk Test for Standardized Residuals:")
print(f"Test statistic: {shapiro_test.statistic:.4f}")
print(f"P-value: {shapiro_test.pvalue:.4f}")
if shapiro_test.pvalue > 0.05:
    print("Residuals appear normally distributed (fail to reject H0).")
else:
    print("Residuals are NOT normally distributed (reject H0).")


# Compute VIF for multicollinearity

X_vif = add_constant(df_reg[['Temp_Anomaly_C', 'CO2_PPM']])
vif_data = pd.DataFrame({
    'Variable': X_vif.columns,
    'VIF': [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
})
print("\nVariance Inflation Factors (VIF):")
print(vif_data)
