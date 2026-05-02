import pandas as pd
import os

def main():
    # Paths
    input_file = r'c:\Users\mduy\source\repos\datathon2026\data\datathon-2026-round-1\analytical\sales.csv'
    output_file = r'c:\Users\mduy\source\repos\datathon2026\submission_scaled.csv'

    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        return

    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file)
    df['Date'] = pd.to_datetime(df['Date'])

    # Filter data: 2021-01-01 to 2022-07-01
    start_date = '2021-01-01'
    end_date = '2022-07-01'
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    df_filtered = df[mask].copy()
    
    print(f"Filtered {len(df_filtered)} rows from {start_date} to {end_date}.")

    # Shift years: 2021 -> 2023, 2022 -> 2024 (add 2 years)
    # We use DateOffset to handle calendar months/years correctly
    df_filtered['TargetDate'] = df_filtered['Date'] + pd.DateOffset(years=2)

    # Multiply Revenue and COGS by 1.25
    df_filtered['Revenue'] = df_filtered['Revenue'] * 1.3
    df_filtered['COGS'] = df_filtered['COGS'] * 1.3

    # Create a continuous target range to ensure no gaps (like 2024-02-29)
    target_start = pd.to_datetime('2023-01-01')
    target_end = pd.to_datetime('2024-07-01')
    target_range = pd.date_range(start=target_start, end=target_end, freq='D')
    
    final_df = pd.DataFrame({'Date': target_range})
    
    # Merge filtered data into the continuous range
    final_df = final_df.merge(
        df_filtered[['TargetDate', 'Revenue', 'COGS']], 
        left_on='Date', 
        right_on='TargetDate', 
        how='left'
    )

    # Handle missing values (like Leap Day 2024-02-29 which wasn't in 2022-02-29 as it didn't exist)
    missing_count = final_df['Revenue'].isna().sum()
    if missing_count > 0:
        print(f"Filling {missing_count} missing values (e.g., leap days) via interpolation...")
        final_df['Revenue'] = final_df['Revenue'].interpolate()
        final_df['COGS'] = final_df['COGS'].interpolate()

    # Final cleanup
    final_df = final_df[['Date', 'Revenue', 'COGS']]
    final_df['Date'] = final_df['Date'].dt.strftime('%Y-%m-%d')

    # Save to CSV
    final_df.to_csv(output_file, index=False)
    print(f"Successfully saved {len(final_df)} rows to {output_file}")

if __name__ == "__main__":
    main()
