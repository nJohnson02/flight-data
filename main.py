import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_flight_data(csv_file):
    # Load CSV file with proper handling of delimiters
    df = pd.read_csv(csv_file, skiprows=1, delimiter=',', engine='python')
    
    # Print column names for debugging
    print("Columns in CSV (before stripping spaces):", df.columns.tolist())
    
    # Clean column names: strip spaces, lowercase, and remove special characters
    df.columns = df.columns.str.strip().str.lower()
    df = df.loc[:, ~df.columns.duplicated()]  # Remove duplicate columns
    
    # Explicitly rename `#yyy-mm-dd` to `date` and `hh:mm:ss` to `time`
    column_mapping = {
        '#yyy-mm-dd': 'date',
        'hh:mm:ss': 'time',
        'ft msl': 'altitude'
    }
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
    
    # Print cleaned column names
    print("Columns in CSV (after stripping spaces and deduplication):", df.columns.tolist())
    
    # Ensure required columns exist before proceeding
    required_columns = ['date', 'time', 'altitude']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print("Error: Missing required columns:", missing_columns, "Available columns:", df.columns.tolist())
        return
    
    # Convert time to datetime with explicit format handling
    try:
        df['time'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    except Exception as e:
        print("Error parsing datetime with explicit format, falling back to automatic parsing:", e)
        df['time'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
    
    # Ensure numeric columns are cleaned before conversion
    df['altitude'] = df['altitude'].astype(str).str.replace(r'[^0-9.-]', '', regex=True)
    df['altitude'] = pd.to_numeric(df['altitude'], errors='coerce')
    
    # Drop NaN values
    df = df.dropna(subset=['time', 'altitude'])
    
    # Identify touchdown point by detecting when altitude stabilizes after descent
    df['altitude_change'] = df['altitude'].diff()
    touchdown_idx = None
    
    for i in range(1, len(df) - 5):  # Looking ahead to confirm stabilization
        if df['altitude_change'].iloc[i] < -5 and df['altitude_change'].iloc[i+1:i+5].abs().max() < 2:
            touchdown_idx = i
            break
    
    touchdown_time = df.loc[touchdown_idx, 'time'] if touchdown_idx is not None else None
    
    # Plot data
    plt.figure(figsize=(10, 6))
    
    plt.plot(df['time'], df['altitude'], label='Altitude (ft MSL)', linestyle='-')
    
    # Mark touchdown point
    if touchdown_time is not None:
        plt.axvline(x=touchdown_time, color='red', linestyle='--', label='Touchdown')
    
    plt.xlabel('Time')
    plt.ylabel('Altitude (ft MSL)')
    plt.title('Flight Data: Altitude with Touchdown Marked')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_flight_data.py <csv_file>")
    else:
        plot_flight_data(sys.argv[1])
