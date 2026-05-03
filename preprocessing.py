import pandas as pd

# Step 1: Load dataset (semicolon separated)
df = pd.read_csv("household_power_consumption.txt", sep=';', low_memory=False)

# Step 2: Combine Date and Time into datetime
df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True, errors='coerce')

# Step 3: Convert Global_active_power to numeric
df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')

# Step 4: Remove missing values (only important columns)
df = df.dropna(subset=['datetime', 'Global_active_power'])

# Step 5: Set datetime as index
df = df.set_index('datetime')

# Step 6: Convert minute data → daily total usage
daily_df = df['Global_active_power'].resample('D').sum()

# Step 7: Convert to proper dataframe
daily_df = daily_df.reset_index()
daily_df.columns = ['date', 'total_daily_usage']

# Step 8: Save cleaned data
daily_df.to_csv("cleaned_daily_data.csv", index=False)

# Step 9: Print output
print("✅ Preprocessing Completed")
print(daily_df.shape)
print(daily_df.head())