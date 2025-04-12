import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import shutil

# Global to capture keypress result
last_key_pressed = None
classification_log = {}


def load_approach_csv(filepath):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"])
    return df.to_dict(orient="records")


def on_key(event):
    global last_key_pressed
    last_key_pressed = event.key
    plt.close()


def plot_approach(data):
    global last_key_pressed
    last_key_pressed = None

    times = [r["Timestamp"] for r in data if pd.notna(r.get("Timestamp"))]
    alts = [float(r["AltGPS"]) for r in data if str(r.get("AltGPS", "")).strip() != ""]
    alt_rate = [((alts[i+1] - alts[i]) / ((times[i+1] - times[i]).total_seconds() or 1)) * 60
                for i in range(len(alts)-1)]
    times_alt_rate = times[:len(alt_rate)]

    start_time = times[0]
    end_time = times[-1]
    landing_altitude = alts[-1]  # Assuming last altitude is at landing

    fig, ax1 = plt.subplots(figsize=(12, 6))
    fig.canvas.mpl_connect('key_press_event', on_key)

    ax1.set_xlabel("Time")
    ax1.set_ylabel("GPS Altitude (ft)", color='tab:blue')
    l1, = ax1.plot(times[:len(alts)], alts, label="GPS Altitude", color='tab:blue')
    ax1.axvline(x=end_time, color='red', linestyle='--', alpha=0.8, label="Landing")
    ax1.axvline(x=start_time, color='blue', linestyle='--', alpha=0.5, label="Approach Start")
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(landing_altitude, landing_altitude + 500)  # Landing altitude at bottom

    ax2 = ax1.twinx()
    ax2.set_ylabel("Altitude Rate (ft/min)", color='tab:green')
    l2, = ax2.plot(times_alt_rate, alt_rate, label="Altitude Rate (ft/min)", linestyle='-', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    ax2.set_ylim(-1000, 200)  # Fixed vertical speed range

    fig.suptitle("Approach Viewer (← back, → = skip, ↑ = good, ↓ = bad, delete = anomaly)")
    fig.legend(handles=[l1, l2], loc="upper right")
    fig.tight_layout()
    plt.grid(True)
    plt.show()

    return last_key_pressed


def view_all_approaches(unclassified_folder):
    def move_file(file_path, target_folder):
        filename = os.path.basename(file_path)
        for folder in [good_folder, bad_folder, anomalies_folder, skipped_folder]:
            existing = os.path.join(folder, filename)
            if os.path.exists(existing):
                os.remove(existing)
        dest = os.path.join(target_folder, filename)
        shutil.move(file_path, dest)
        return dest

    def unclassify_file(filename):
        for folder in [good_folder, bad_folder, anomalies_folder, skipped_folder]:
            path = os.path.join(folder, filename)
            if os.path.exists(path):
                dest = os.path.join(unclassified_folder, filename)
                shutil.move(path, dest)
                return dest
        return None

    base_folder = os.path.dirname(unclassified_folder)
    good_folder = os.path.join(base_folder, "good")
    bad_folder = os.path.join(base_folder, "bad")
    anomalies_folder = os.path.join(base_folder, "anomalies")
    skipped_folder = os.path.join(base_folder, "skipped")

    os.makedirs(good_folder, exist_ok=True)
    os.makedirs(bad_folder, exist_ok=True)
    os.makedirs(anomalies_folder, exist_ok=True)
    os.makedirs(skipped_folder, exist_ok=True)

    # Initialize file list with classification=None
    file_list = []
    for file in sorted(glob.glob(os.path.join(unclassified_folder, "*.csv"))):
        file_list.append([file, None])

    i = 0
    while 0 <= i < len(file_list):
        file_path, classification = file_list[i]
        filename = os.path.basename(file_path)

        print(f"Showing approach {i + 1} of {len(file_list)}: {filename}")
        data = load_approach_csv(file_path)
        key = plot_approach(data)

        if key == 'left' and i > 0:
            prev_file, prev_class = file_list[i - 1]
            prev_name = os.path.basename(prev_file)
            new_path = unclassify_file(prev_name)
            if new_path:
                file_list[i - 1][0] = new_path
                file_list[i - 1][1] = None
            i -= 1
        elif key == 'right':
            dest = move_file(file_path, skipped_folder)
            file_list[i][0] = dest
            file_list[i][1] = "skipped"
            i += 1
        elif key == 'up':
            dest = move_file(file_path, good_folder)
            file_list[i][0] = dest
            file_list[i][1] = "good"
            i += 1
        elif key == 'down':
            dest = move_file(file_path, bad_folder)
            file_list[i][0] = dest
            file_list[i][1] = "bad"
            i += 1
        elif key == 'delete':
            dest = move_file(file_path, anomalies_folder)
            file_list[i][0] = dest
            file_list[i][1] = "anomalies"
            i += 1
        else:
            print("No action taken\n")


if __name__ == "__main__":
    view_all_approaches("manual_classification/unclassified")