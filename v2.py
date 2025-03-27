import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import numpy as np


def load_csv(csv_path):
    with open(csv_path, "r") as f:
        lines = f.readlines()

    header_line_idx = None
    for i, line in enumerate(lines):
        if "AltMSL" in line and "VSpd" in line:
            header_line_idx = i
            break

    if header_line_idx is None:
        raise ValueError("No valid header found in CSV.")

    df = pd.read_csv(csv_path, skiprows=header_line_idx)
    df.columns = df.columns.str.strip()

    df["Timestamp"] = pd.to_datetime(
        df["Lcl Date"].astype(str).str.strip() + " " + df["Lcl Time"].astype(str).str.strip(),
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce"
    )

    df = df.dropna(subset=["Timestamp"])

    return df.to_dict(orient="records")


def detect_approaches(records, order=150, min_climb=500, lookback_seconds=180, field_elevation=5045, elevation_tolerance=500, approach_lead_time=60):
    altitudes = np.array([float(r["AltMSL"]) for r in records if str(r["AltMSL"]).strip() != ""])
    timestamps = [r["Timestamp"] for r in records if str(r["AltMSL"]).strip() != ""]

    minima_indices = argrelextrema(altitudes, np.less, order=order)[0]

    approaches = []
    for idx in minima_indices:
        t_min = timestamps[idx]
        a_min = altitudes[idx]

        if abs(a_min - field_elevation) > elevation_tolerance:
            continue

        lookback_start_time = t_min - pd.Timedelta(seconds=lookback_seconds)
        for j in range(max(0, idx - order * 3), idx):
            if timestamps[j] < lookback_start_time:
                continue
            if altitudes[j] >= a_min + min_climb:
                approach = {
                    "start": t_min - pd.Timedelta(seconds=approach_lead_time),
                    "end": t_min
                }
                approaches.append(approach)
                break

    return approaches


def plot(records):
    times = [r["Timestamp"] for r in records if "Timestamp" in r and pd.notna(r["Timestamp"])]
    alts = [float(r["AltMSL"]) for r in records if "AltMSL" in r and str(r["AltMSL"]).strip() != ""]

    approaches = detect_approaches(records)

    plt.figure(figsize=(12, 6))
    plt.plot(times[:len(alts)], alts, label="Altitude (MSL)", linewidth=2)

    for approach in approaches:
        plt.axvline(x=approach["end"], color='red', linestyle='--', alpha=0.8)
        plt.axvline(x=approach["start"], color='blue', linestyle='--', alpha=0.5)

    plt.xlabel("Time")
    plt.ylabel("Altitude (ft MSL)")
    plt.title("Altitude vs Time with Detected Landings and Approaches")
    plt.grid(True)
    plt.legend(["Altitude", "Landing", "Approach Start"])
    plt.tight_layout()
    plt.show()


# Example usage
csv_path = "/media/nathan/bulk-storage/prescott-flight-logs/log_240311_133613_KPRC.csv"
records = load_csv(csv_path)
plot(records)
