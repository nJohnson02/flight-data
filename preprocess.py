import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import numpy as np
import os
import glob


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


def detect_approaches(records, order=150, min_climb=500, lookback_seconds=180, field_elevation=5045, elevation_tolerance=500, approach_duration=60):
    altitudes = np.array([float(r["AltMSL"]) for r in records if str(r["AltMSL"]).strip() != ""])
    timestamps = [r["Timestamp"] for r in records if str(r["AltMSL"]).strip() != ""]
    all_records = [r for r in records if str(r["AltMSL"]).strip() != ""]

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
                start_time = t_min - pd.Timedelta(seconds=approach_duration)
                approach_data = [r for r in all_records if start_time <= r["Timestamp"] <= t_min]
                approach = {
                    "start": start_time,
                    "end": t_min,
                    "data": approach_data
                }
                approaches.append(approach)
                break

    return approaches


def export_approaches_to_csv(approaches, output_dir="approaches_export"):
    os.makedirs(output_dir, exist_ok=True)
    for approach in approaches:
        df = pd.DataFrame(approach["data"])
        timestamp_str = approach["start"].strftime("%Y%m%d_%H%M%S")
        filename = f"approach_{timestamp_str}.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
    print(f"Exported {len(approaches)} approaches to '{output_dir}/'")


def process_all_logs(logs_dir):
    all_files = glob.glob(os.path.join(logs_dir, "*.csv"))
    total_approaches = 0
    for filepath in all_files:
        try:
            print(f"Processing {filepath}...")
            records = load_csv(filepath)
            approaches = detect_approaches(records)
            #plot_all(approaches)
            export_approaches_to_csv(approaches)
            total_approaches += len(approaches)
        except Exception as e:
            print(f"Failed to process {filepath}: {e}")
    print(f"Total approaches exported: {total_approaches}")


def plot(approach):
    data = approach["data"]
    times = [r["Timestamp"] for r in data if pd.notna(r.get("Timestamp")) and approach["start"] <= r["Timestamp"] <= approach["end"]]
    alts = [float(r["AltMSL"]) for r in data if pd.notna(r.get("Timestamp")) and approach["start"] <= r["Timestamp"] <= approach["end"] and str(r["AltMSL"]).strip() != ""]
    vspeeds = [float(r["VSpd"]) for r in data if pd.notna(r.get("Timestamp")) and approach["start"] <= r["Timestamp"] <= approach["end"] and str(r.get("VSpd", "")).strip() != ""]
    ias = [float(r["IAS"]) for r in data if pd.notna(r.get("Timestamp")) and approach["start"] <= r["Timestamp"] <= approach["end"] and str(r.get("IAS", "")).strip() != ""]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel("Time")
    ax1.set_ylabel("Altitude (ft MSL)", color='tab:blue')
    l1, = ax1.plot(times[:len(alts)], alts, label="Altitude (MSL)", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    l2 = ax1.axvline(x=approach["end"], color='red', linestyle='--', alpha=0.8, label="Landing")
    l3 = ax1.axvline(x=approach["start"], color='blue', linestyle='--', alpha=0.5, label="Approach Start")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Vertical Speed / IAS", color='tab:gray')
    l4, = ax2.plot(times[:len(vspeeds)], vspeeds, label="Vertical Speed", linestyle='-', color='tab:orange')
    l5, = ax2.plot(times[:len(ias)], ias, label="IAS", linestyle='-', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:gray')

    fig.suptitle("Detected Approach Segment")
    fig.legend(handles=[l1, l4, l5, l2, l3], loc="upper right")
    fig.tight_layout()
    plt.grid(True)
    plt.show()


def plot_all(approaches):
    num_approaches = len(approaches)
    fig, axes = plt.subplots(num_approaches, 1, figsize=(12, 6 * num_approaches), sharex=False)

    if num_approaches == 1:
        axes = [axes]

    for i, (approach, ax1) in enumerate(zip(approaches, axes)):
        data = approach["data"]
        times = [r["Timestamp"] for r in data if pd.notna(r.get("Timestamp")) and approach["start"] <= r["Timestamp"] <= approach["end"]]
        alts = [float(r["AltMSL"]) for r in data if pd.notna(r.get("Timestamp")) and approach["start"] <= r["Timestamp"] <= approach["end"] and str(r["AltMSL"]).strip() != ""]
        vspeeds = [float(r["VSpd"]) for r in data if pd.notna(r.get("Timestamp")) and approach["start"] <= r["Timestamp"] <= approach["end"] and str(r.get("VSpd", "")).strip() != ""]
        ias = [float(r["IAS"]) for r in data if pd.notna(r.get("Timestamp")) and approach["start"] <= r["Timestamp"] <= approach["end"] and str(r.get("IAS", "")).strip() != ""]

        ax1.set_title(f"Approach Segment {i + 1}")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Altitude (ft MSL)", color='tab:blue')
        ax1.plot(times[:len(alts)], alts, label="Altitude (MSL)", color='tab:blue')
        ax1.axvline(x=approach["end"], color='red', linestyle='--', alpha=0.8, label="Landing")
        ax1.axvline(x=approach["start"], color='blue', linestyle='--', alpha=0.5, label="Approach Start")
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel("Vertical Speed / IAS", color='tab:gray')
        ax2.plot(times[:len(vspeeds)], vspeeds, label="Vertical Speed", linestyle='-', color='tab:orange')
        ax2.plot(times[:len(ias)], ias, label="IAS", linestyle='-', color='tab:green')
        ax2.tick_params(axis='y', labelcolor='tab:gray')

        if i == 0:
            fig.legend(loc="upper right")

    fig.tight_layout()
    plt.grid(True)
    plt.show()


# Process all flight logs in directory
process_all_logs("/media/nathan/bulk-storage/prescott-flight-logs")
