"""
Project: VLIZ Pi-10 processing pipeline
Module:  VLIZ_Pi-10_processing.py

Metadata
--------
- Authors: Jonas Mortelmans <jonas.mortelmans@vliz.be>, Wout Decrop <wout.decrop@vliz.be>
- Created: 2025-10-03
- Updated: 2025-20-04
- Version: 1.0.0
- Documentation: Mortelmans J., Decrop W., Heynderickx H., Cattrijsse A., Depaepe M., Van Walraeven L., Scott J., Van Oevelen D., Deneudt K., Muniz C. (2025, submitted). High-throughput image classification and morphometry though the Pi-10 imaging pipeline
- Source: https://github.com/ai4os-hub/phyto-plankton-classification/blob/PI10/PI10/VLIZ-Pi-10_processing.py
"""


# === LIBRARIES ===
import os
from pathlib import Path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

# Point the installed package to this repo's local config before importing planktonclass.
REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ["PLANKTONCLASS_CONFIG"] = str(REPO_ROOT / "config.yaml")
os.environ["planktonclass_CONFIG"] = str(REPO_ROOT / "config.yaml")

import smtplib
from email.mime.text import MIMEText
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import shutil
import tarfile
import pandas as pd
import subprocess
import time
import random
import json
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage import measure, morphology
from tensorflow.keras.models import load_model
from planktonclass import paths as plk_paths, utils
from planktonclass.test_utils import predict
from planktonclass.data_utils import load_class_names
import datetime
import threading
import csv
from dotenv import load_dotenv

print("start")

last_summary_date = None  # will track the last date email was sent
last_afternoon_summary_sent_day = None  # for 15:00 status update

# === STORE LOGS ===
log_file_path = "/paths/PI10/logs/processing_times.csv"



#=== MAILING ===
load_dotenv(dotenv_path=".env")
EMAIL_SETTINGS = {
    'smtp_server': os.getenv('SMTP_SERVER'),
    'smtp_port': int(os.getenv('SMTP_PORT')),
    'sender_email': os.getenv('SENDER_EMAIL'),
    'sender_password': os.getenv('SENDER_PASSWORD'),
    'recipients': [email.strip() for email in os.getenv('EMAIL_RECIPIENTS').split(',')]
}
daily_tar_reports = []  # Stores dicts with tar_name, quarantined, quarantine_reason, quarantine_path, status_log


#test

def email_scheduler():
    global last_afternoon_summary_sent_day
    while True:
        now = datetime.datetime.now()
        current_date = now.date()

        # Send summary at exactly 15:00 once per day
        if (now.hour == 11 and now.minute == 0
                and last_afternoon_summary_sent_day != current_date):
            send_daily_summary_email(now.strftime('%Y-%m-%d'), daily_tar_reports)
            last_afternoon_summary_sent_day = current_date

        time.sleep(60)  # check every minute

def send_daily_summary_email(summary_date, report_data):
    import math
    global source_dir

    subject = f"[PI10] Daily Summary - {summary_date}"
    all_tar_files = list(source_dir.glob("*.tar"))
    tar_stems = {tar.stem for tar in all_tar_files}

    # === Count current files ===
    preview_root = Path("/paths/plankton-imager-10/not_processed/previews")
    current_counts = {
        "tar": len(all_tar_files),
        "gpstag": len(list(source_dir.glob("*_gpstag.csv"))),
        "predictions": len(list(source_dir.glob("*_predictions_relative.json"))),
        "image_props": len(list(source_dir.glob("*_image_properties.csv"))),
        "topspecies": len(list(source_dir.glob("*_topspecies.csv"))),
        "hitsmisses": len(list(source_dir.glob("*_hitsmisses.txt"))),
        "backgrounds": len(list(source_dir.glob("*_Background.tif"))),
        "previews": len(list(preview_root.glob("*_preview"))),
    }

    # === Load yesterday's counts from file ===
    delta_file = Path("daily_tar_count.json")
    yesterday_counts = {k: 0 for k in current_counts}
    if delta_file.exists():
        try:
            with open(delta_file, 'r') as f:
                yesterday_counts.update(json.load(f))
        except Exception as e:
            print(f"⚠️ Could not read yesterday's count: {e}")

    # === Calculate deltas ===
    deltas = {k: current_counts[k] - yesterday_counts.get(k, 0) for k in current_counts}
    tar_delta = deltas["tar"]

    # === Save today’s counts for tomorrow ===
    try:
        with open(delta_file, 'w') as f:
            json.dump(current_counts, f)
    except Exception as e:
        print(f"⚠️ Could not write today's count: {e}")

    # === Calculate to-do (missing output per TAR) ===
    required_outputs = {
        "_gpstag.csv": ("GPS data", 0.5),                   # 30 mins per file
        "_hitsmisses.txt": ("Hits/Misses", 10 / 3600),      # 10 sec per file
        "_Background.tif": ("Background.tif", 10 / 3600),   # 10 sec per file
        "_predictions_relative.json": ("Predictions (JSON)", 3),  # 3 hours per file
        "_image_properties.csv": ("Image Properties (CSV)", 0.5), # 30 mins per file
        "_topspecies.csv": ("Top Species CSV", 2 / 60),     # 2 mins per file
    }

    todo_counts = {}
    raw_time_estimations = {}
    formatted_time_estimations = {}

    for suffix, (label, per_file_hours) in required_outputs.items():
        count = sum(not (source_dir / f"{stem}{suffix}").exists() for stem in tar_stems)
        todo_counts[label] = count
        total_hours = count * per_file_hours
        raw_time_estimations[label] = total_hours

        # Format time
        if total_hours >= 24:
            formatted_time_estimations[label] = f"{round(total_hours / 24, 2)} day(s)"
        else:
            h = int(total_hours)
            m = round((total_hours - h) * 60)
            formatted_time_estimations[label] = f"{h}h {m}min"

    total_time = sum(raw_time_estimations.values())
    if total_time >= 24:
        total_time_str = f"{round(total_time / 24, 2)} day(s)"
    else:
        th = int(total_time)
        tm = round((total_time - th) * 60)
        total_time_str = f"{th}h {tm}min"

    # === Build email body ===
    body_lines = []
    body_lines.append(f"🆕 **{abs(tar_delta)} TARs {'extra' if tar_delta >= 0 else 'less'} compared to yesterday**")
    body_lines.append("=" * 60)
    body_lines.append("")

    body_lines.append(f"**Summary for {summary_date}**")
    body_lines.append(f"TARs entirely processed today: {len(report_data)}")
    body_lines.append("")

    body_lines.append("**Folder Totals vs Yesterday:**")
    for key, label in [
        ("tar", "TAR files"),
        ("gpstag", "GPS data (gpstag.csv)"),
        ("predictions", "Predictions (JSON)"),
        ("image_props", "Image Properties (CSV)"),
        ("topspecies", "Top Species CSV"),
        ("hitsmisses", "Hits/Misses TXT"),
        ("backgrounds", "Background.tif"),
        ("previews", "Preview folders"),
    ]:
        delta = deltas[key]
        sign = "+" if delta >= 0 else "-"
        body_lines.append(f"- {label}: {current_counts[key]} ({sign}{abs(delta)})")

    body_lines.append("")
    body_lines.append("**To-do by output module (missing files):**")
    for label in todo_counts:
        count = todo_counts[label]
        formatted_time = formatted_time_estimations[label]
        body_lines.append(f"- {label}: {count} files missing ({formatted_time})")

    body_lines.append("")
    body_lines.append(f"**Total estimated processing time left:** {total_time_str}")

    # === Send email ===
    msg = MIMEText("\n".join(body_lines))
    msg['Subject'] = subject
    msg['From'] = EMAIL_SETTINGS['sender_email']
    msg['To'] = ", ".join(EMAIL_SETTINGS['recipients'])

    try:
        with smtplib.SMTP(EMAIL_SETTINGS['smtp_server'], EMAIL_SETTINGS['smtp_port']) as server:
            server.starttls()
            server.login(EMAIL_SETTINGS['sender_email'], EMAIL_SETTINGS['sender_password'])
            server.sendmail(msg['From'], EMAIL_SETTINGS['recipients'], msg.as_string())
        print(f"📧 Daily summary email sent for {summary_date}")
    except Exception as e:
        print(f"❌ Failed to send daily summary email: {e}")



#LOG TIME OF EACH STEP
def init_log_file():
    """Initialize the CSV log file with headers."""
    headers = [
        "TAR Name",
        "Copy TAR to working directory",
        "Untar",
        "Extract hitsmisses.txt",
        "Count images",
        "Create preview images",
        "Early preview classification",
        "Copy Background.tif",
        "Extract and save EXIF metadata",
        "Classification and morphology extraction",
        "Generate top species CSV",
        "Per-minute bio metrics",
        "Total pipeline time (h)",
        "Number of images in TAR",
        "Model used",
        "Logged at"
    ]

    if not Path(log_file_path).exists():
        with open(log_file_path, "w", newline="") as log_file:
            writer = csv.writer(log_file)
            writer.writerow(headers)
        print("⚙ Initialized new processing time log file.")
    else:
        print("⚙ Log file already exists, appending new entries.")

# for the logfiles; these are the headers
step_names = [
    "Copy TAR to working directory",
    "Untar",
    "Extract hitsmisses.txt",
    "Count images",
    "Create preview images",
    "Early preview classification",
    "Copy Background.tif",
    "Extract and save EXIF metadata",
    "Classification and morphology extraction",
    "Generate top species CSV",
    "Per-minute bio metrics"
]

def log_time_to_file(tar_name, times_dict, num_images):
    total_hours = sum(times_dict.values()) / 3600.0
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row = [tar_name] + [times_dict.get(name, 0.0) for name in step_names] \
          + [total_hours, num_images, TIMESTAMP, timestamp]

    with open(log_file_path, "a", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(row)

    #print(f"✅ Logged times for {tar_name} to file (total {total_hours:.2f} h).")


def track_time(start_time, module_name):
    """Calculate elapsed time and return the time taken."""
    elapsed_time = time.time() - start_time
    return elapsed_time




# === SETUP ===
source_dir = Path("/paths/plankton-imager-10/processed")
work_dir = Path("/paths/PI10_tempUntarred")

quarantine_bubbles_dir = Path("/paths/plankton-imager-10/processed/quarantine-bubbles")
quarantine_bubbles_dir.mkdir(parents=True, exist_ok=True)

quarantine_hitsmiss_dir = Path("/paths/plankton-imager-10/processed/quarantine-hitsmisses")
quarantine_hitsmiss_dir.mkdir(parents=True, exist_ok=True)


os.makedirs(work_dir, exist_ok=True)
os.chdir(work_dir)
email_thread = threading.Thread(target=email_scheduler, daemon=True)
email_thread.start()
init_log_file()  # Initialize log file right after setup


paths = {
    'tarred': work_dir / "data/tarred",
    'untarred': work_dir / "data/untarred",
    'output': work_dir / "output",
    'hitsmisses': work_dir / "data/hitsmisses"
}

for path in paths.values():
    path.mkdir(parents=True, exist_ok=True)

# Classification model setup
TIMESTAMP = '2025-09-22_153855-doe'
MODEL_NAME = 'final_model.h5'
TOP_K = 3
plk_paths.timestamp = TIMESTAMP
class_names = load_class_names(splits_dir=plk_paths.get_ts_splits_dir())
model = load_model(os.path.join(plk_paths.get_checkpoints_dir(), MODEL_NAME),
                   custom_objects=utils.get_custom_objects())
with open(os.path.join(plk_paths.get_conf_dir(), 'conf.json')) as f:
    conf = json.load(f)


# === HELPER FUNCTIONS ===
def outputs_exist_for_tar(tar_file):
    stem = tar_file.stem
    required = [
        source_dir / f"{stem}_gpstag.csv",
        source_dir / f"{stem}_hitsmisses.txt",
        source_dir / f"{stem}_Background.tif",
        source_dir / f"{stem}_predictions_relative.json",
        source_dir / f"{stem}_image_properties.csv",
        source_dir / f"{stem}_topspecies.csv",
         source_dir / f"{stem}_bio-metrics.csv"
    ]
    preview_dir = Path(r"/paths/plankton-imager-10/not_processed/previews") / f"{stem}_preview"

    if not preview_dir.exists() or not any(preview_dir.glob("*.tif")):
        return False

    for f in required:
        if not f.exists():
            return False
    return True


def get_new_tar_files(source_dir):
    all_tar = list(source_dir.glob("*.tar"))

    # combine both quarantine folders
    quarantine_stems = set()
    quarantine_stems.update({tar.stem for tar in quarantine_bubbles_dir.glob("*.tar")})
    quarantine_stems.update({tar.stem for tar in quarantine_hitsmiss_dir.glob("*.tar")})

    done_stems = {p.stem for p in source_dir.glob("*.done")}

    new_files = []
    for tar in all_tar:
        # skip if in quarantine
        if tar.stem in quarantine_stems:
            continue
        # skip if already marked done
        if tar.stem in done_stems:
            continue

        # check outputs
        outputs_to_check = [
            "_gpstag.csv",
            "_hitsmisses.txt",
            "_Background.tif",
            "_predictions_relative.json",
            "_image_properties.csv",
            "_topspecies.csv",
            "_bio-metrics.csv"
        ]
        preview_dir = Path(r"/paths/plankton-imager-10/not_processed/previews")
        missing_output = False

        for suffix in outputs_to_check:
            expected = source_dir / f"{tar.stem}{suffix}"
            if not expected.exists():
                missing_output = True
                break

        preview_path = preview_dir / f"{tar.stem}_preview"
        if not preview_path.exists() or not any(preview_path.glob("*.tif")):
            missing_output = True

        if missing_output:
            new_files.append(tar)

    return new_files


from time import time as timer

def clear_untarred_dir(dir_path):
    start_time = timer()  # Start timing
    print(f"⚙ Clear and created local directories")
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True)  # Recreate the directory

    elapsed_time = timer() - start_time  # Calculate elapsed time
    print(f"       ✅ Done in {elapsed_time:.2f} seconds.")
    return elapsed_time  # Return the time taken


def extract_tar(tar_path, extract_to):
    start_time = timer()  # Start timing
    print(f"⚙ Untarring {tar_path.name}...")

    with tarfile.open(tar_path) as tar:
        tar.extractall(path=extract_to)  # Extract the TAR file

    elapsed_time = timer() - start_time  # Calculate elapsed time
    print(f"       ✅ Done in {elapsed_time:.2f} seconds.")
    return elapsed_time  # Return the time taken


def count_images_in_tar(extract_dir, tar_file):
    """Count the number of .tif images in the extracted directory."""
    print(f"⚙ Counting images in {tar_file.name}...")
    tif_files = list(extract_dir.rglob("*.tif"))
    print(f"       ✅ Found {len(tif_files)} .tif files")
    return len(tif_files)

def copy_background_tif(extract_dir, dest_path):
    start = timer()
    print("⚙ Copying Background..")

    for root, _, files in os.walk(extract_dir):
        for f in files:
            if f == "Background.tif":
                full_path = os.path.join(root, f)
                if not dest_path.exists():
                    shutil.copy(full_path, dest_path)
                elapsed_time = timer() - start
                print(f"       ✅ Done in {elapsed_time:.2f} seconds.")
                return  # exit after success

    # If loop finishes without finding the file
    elapsed_time = timer() - start
    print(f"       ⚠️ Background.tif not found")



def extract_hitsmisses(tar_path, output_file, tar_file, status_log):
    start = timer()
    print("⚙ Fetching hits and misses...")

    with tarfile.open(tar_path) as tar:
        hits_file = next((m for m in tar.getmembers() if "hitsmisses.txt" in m.name.lower()), None)
        if hits_file:
            f = tar.extractfile(hits_file)
            df = pd.read_csv(f, header=None)
            df.columns = ['hits', 'misses']
            df['minute'] = range(len(df))
            df['tar_source'] = tar_path.stem
            df.to_csv(output_file, index=False)

            # Calculate RaisingFactor (sum of hits and misses divided by hits)
            df['RaisingFactor'] = df['hits']/(df['hits'] + df['misses'])

            #  Check row count
            if len(df) < 10:  # should be at least 10 rows
                quarantine_target = quarantine_hitsmiss_dir / tar_file.name
                shutil.move(str(tar_file), str(quarantine_target))
                status_log.append(f"Moved to quarantine_hitsmiss (<10 rows)")
                print(f"       🚨 Quarantined {tar_file.name}: hitsmisses had only {len(df)} rows, moved to quarantine")

                # Optionally, clear the hitsmisses.txt if needed
                try:
                    if output_file.exists():
                        os.remove(output_file)
                        status_log.append(f"Removed hitsmisses.txt due to quarantine")
                except Exception as e:
                    status_log.append(f"⚠️ Failed to remove hitsmisses.txt: {e}")

                return False  # signal quarantine
    elapsed_time = timer() - start
    print(f"       ✅ Done in {elapsed_time:.2f} seconds.")
    return True



def create_preview_images(extract_dir, preview_dir, tar_name, n=200):
    start = timer()
    print(f"⚙ Saving preview image....       ")
    tif_files = list(extract_dir.rglob("*.tif"))
    if len(tif_files) < n:
        print(f"       ⚠️ Not enough .tif files in {extract_dir}")
        return
    selected = random.sample(tif_files, n)
    preview_path = preview_dir / f"{tar_name}_preview"
    preview_path.mkdir(exist_ok=True)

    from concurrent.futures import ThreadPoolExecutor
    def copy_preview(tif):
        new_name = tif.stem + "_preview.tif"
        shutil.copy(tif, preview_path / new_name)
    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(copy_preview, selected)
    elapsed_time = timer() - start
    print(f"       ✅ Done in {elapsed_time:.2f} seconds.")



def extract_metadata(tif_paths, tar_source, batch_size=200):
    print("⚙ Extracting EXIF metadata in batch...")

    exiftool_path = r"\paths\PI10\exiftool-13.31_64\exiftool.exe"
    all_data = []
    tif_paths = list(tif_paths)
    n_batches = (len(tif_paths) + batch_size - 1) // batch_size

    total_start_time = time.time()

    for batch_idx in range(n_batches):
        batch = tif_paths[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        args = [
            exiftool_path,
            '-GPSLatitude',
            '-GPSLongitude',
            '-FileModifyDate',
            '-DateTimeOriginal',
            '-CreateDate',
            '-ModifyDate',
            '-n',
            '-api', 'QuickTimeUTC',
            '-api', 'ExifToolVersion=12.31'
        ] + [str(p) for p in batch]

        try:
            result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                print(f"       ❌ Exiftool error (batch {batch_idx}): {result.stderr}")
                continue

            data, exif = [], {}
            for line in result.stdout.strip().splitlines():
                if line.startswith("======== "):
                    if exif:
                        data.append(exif)
                    exif = {"SourceFile": line.replace("======== ", "").strip()}
                elif ":" in line:
                    k, v = line.split(":", 1)
                    k = k.strip().replace(" ", "")  # normalize key
                    exif[k] = v.strip()
            if exif:
                data.append(exif)

            all_data.extend(data)

        except Exception as e:
            print(f"Skipping due to error: {e}")
            continue

    print(f"       ✅ Done in {time.time()-total_start_time:.2f} seconds ({len(all_data)} rows)")
    df = pd.DataFrame(all_data)
    if not df.empty:
        df["tar_source"] = tar_source
        if "SourceFile" in df.columns:
            df["tif_name"] = df["SourceFile"].apply(lambda x: os.path.basename(str(x)))
    return df

    #for batch_idx in tqdm(range(n_batches), desc="ExifTool Batches", unit="batch"): #with ocuntrer
    for batch_idx in range(n_batches):
        batch = tif_paths[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        args = [
                   exiftool_path,
                   '-GPSLatitude',
                   '-GPSLongitude',
                   '-FileModifyDate',
                   '-DateTimeOriginal',  # add this
                   '-n',
                   '-api', 'QuickTimeUTC',
                   '-api', 'ExifToolVersion=12.31'
               ] + [str(p) for p in batch]

        try:
            result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                print(f"       ❌ Exiftool error (batch {batch_idx}): {result.stderr}")
                continue

            data = []
            exif = {}
            for line in result.stdout.strip().splitlines():
                if line.startswith("======== "):
                    if exif:
                        data.append(exif)
                    exif = {"SourceFile": line.replace("======== ", "").strip()}
                elif ":" in line:
                    k, v = line.split(":", 1)
                    exif[k.strip()] = v.strip()
            if exif:
                data.append(exif)

            all_data.extend(data)

        except Exception:
            continue  # fully silent


    total_elapsed_time = time.time() - total_start_time  # Calculate total elapsed time for EXIF extraction
    print(f"       ✅ Done in {total_elapsed_time:.2f} seconds ({len(all_data)-2} of {len(tif_paths)-2} )")

    df = pd.DataFrame(all_data)

    if not df.empty:
        df["tar_source"] = tar_source
        df["tif_name"] = df["SourceFile"].apply(lambda x: os.path.basename(x))

        if 'DateTimeOriginal' in df.columns:
            df['FileModifyDate_parsed'] = pd.to_datetime(
                df['DateTimeOriginal'], errors='coerce'
            ).dt.strftime('%-m/%-d/%Y  %-I:%M:%S %p')
    return df



def write_exif_csvs(df, tar_name, output_dir, backup_dir):
    # Ensure tif_name exists
    if "tif_name" not in df.columns and "SourceFile" in df.columns:
        df["tif_name"] = df["SourceFile"].apply(lambda x: os.path.basename(str(x)))

    # Accept multiple possible timestamp fields
    time_keys = ["DateTimeOriginal", "FileModifyDate", "CreateDate", "ModifyDate"]

    cols = []
    if "SourceFile" in df.columns: cols.append("SourceFile")
    cols.append("tif_name")

    for c in ["GPSLatitude", "GPSLongitude"] + time_keys:
        if c in df.columns:
            cols.append(c)

    df = df[cols]

    outname = f"{tar_name}_gpstag.csv"
    (output_dir / outname).parent.mkdir(parents=True, exist_ok=True)
    (backup_dir / outname).parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / outname, index=False)
    df.to_csv(backup_dir / outname, index=False)
    #print(f"✅ Saved EXIF CSV with GPS/timestamps: {outname}")



def getImageRegionList(filename):
    image = imread(filename)
    if image.ndim == 3:
        image = rgb2gray(image)
    image_threshold = np.where(image > np.mean(image), 0., 1.0)
    image_dilated = morphology.dilation(image_threshold, np.ones((4, 4)))
    label_list = measure.label(image_dilated)
    label_list = (image_threshold * label_list).astype(int)
    return measure.regionprops(label_list)

def getMaxAreaDict(filename):
    regions = getImageRegionList(filename)
    if not regions:
        return {'object_additional_area': 0}
    r = max(regions, key=lambda x: x.area)
    return {
        'object_additional_diameter_equivalent': r.equivalent_diameter,
        'object_additional_length_minor_axis': r.minor_axis_length,
        'object_additional_length_major_axis': r.major_axis_length,
        'object_additional_eccentricity': r.eccentricity,
        'object_additional_area': r.area,
        'object_additional_perimeter': r.perimeter,
        'object_additional_orientation': r.orientation,
        'object_additional_area_convex': r.convex_area,
        'object_additional_area_filled': r.filled_area,
        'object_additional_box_min_row': r.bbox[0],
        'object_additional_box_max_row': r.bbox[2],
        'object_additional_box_min_col': r.bbox[1],
        'object_additional_box_max_col': r.bbox[3],
        'object_additional_ratio_extent': r.extent,
        'object_additional_ratio_solidity': r.solidity,
        'object_additional_inertia_tensor_eigenvalue1': r.inertia_tensor_eigvals[0],
        'object_additional_inertia_tensor_eigenvalue2': r.inertia_tensor_eigvals[1],
        'object_additional_moments_hu1': r.moments_hu[0],
        'object_additional_moments_hu2': r.moments_hu[1],
        'object_additional_moments_hu3': r.moments_hu[2],
        'object_additional_moments_hu4': r.moments_hu[3],
        'object_additional_moments_hu5': r.moments_hu[4],
        'object_additional_moments_hu6': r.moments_hu[5],
        'object_additional_moments_hu7': r.moments_hu[6],
        'object_additional_euler_number': r.euler_number,
        'object_additional_countcoords': len(r.coords)
    }

def classify_and_extract_regions(tar_file, extract_dir):
    start_time = time.time()
    base_name = tar_file.stem
    json_path = source_dir / f"{base_name}_predictions_relative.json"
    csv_path = source_dir / f"{base_name}_image_properties.csv"
    FILEPATHS = list(extract_dir.rglob("*.tif"))

    # Filter only useful files
    FILEPATHS = [p for p in FILEPATHS if "Background.tif" not in p.name and "FlowCellEdges.tif" not in p.name]

    if not FILEPATHS:
        print(f"⚠️ No valid .tif files in {base_name}, skipping.")
        return

    print(f"⚙ Predicting {len(FILEPATHS)} TIFF files")

    # Run prediction
    pred_lab, pred_prob = predict(model, FILEPATHS, conf, top_K=TOP_K, filemode='local')

    results_json = []
    results_csv = []

    for i, path in enumerate(FILEPATHS):
        rel_path = str(path.relative_to(extract_dir))

        # === JSON prediction ===
        labels = [class_names[pred_lab[i, j]] for j in range(TOP_K)]
        probs = [float(pred_prob[i, j]) for j in range(TOP_K)]
        results_json.append({
            "filepath": rel_path,
            f"top{TOP_K}_labels": labels,
            f"top{TOP_K}_probs": probs
        })

        # === Morphology extraction ===
        try:
            props = getMaxAreaDict(path)
            props["filepath"] = rel_path
            results_csv.append(props)
        except Exception as e:
            print(f"       ❌ Error processing {rel_path}: {e}")

    # Save JSON
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    elapsed_time = time.time() - start_time
    print(f"       ✅ Done in {elapsed_time / 3600:.1f} hours.")

    # Save CSV
    if results_csv:
        pd.DataFrame(results_csv).to_csv(csv_path, index=False)
        #print(f"       ✅ Saved image properties CSV: {csv_path.name}")
    else:
        print(f"       ⚠️No region properties written for {base_name}")
    return pred_lab  # at the end


def generate_topspecies_csv(json_path,
                            upper_threshold=0.95,
                            diff_threshold=0.6,
                            decimals=6):
    print("⚙ Generating top species CSV")

    # Exists?
    if not json_path.exists():
        print(f"       ❌ JSON file not found: {json_path}")
        return

    # Read JSON
    try:
        with open(json_path, "r") as f:
            data_list = json.load(f)
    except Exception as e:
        print(f"       ❌ Failed to read JSON: {e}")
        return

    if not isinstance(data_list, list) or not data_list:
        print("       ⚠️ JSON is empty or invalid.")
        return

    rows = []
    for entry in data_list:
        filepath = entry.get("filepath", "")
        labels   = entry.get("top3_labels", [])
        probs    = entry.get("top3_probs", [])

        # Coerce "labels/probs" in case they're comma-separated strings
        if isinstance(labels, str):
            labels = [s.strip() for s in labels.split(",")]
        if isinstance(probs, str):
            try:
                probs = [float(s.strip()) for s in probs.split(",")]
            except ValueError:
                probs = []

        # Need at least top-2 to compute the margin
        if not filepath or len(labels) < 2 or len(probs) < 2:
            continue

        label = labels[0]
        prob1 = float(probs[0])
        prob2 = float(probs[1])

        # Conditionally append _AI99
        if prob1 > upper_threshold and (prob1 - prob2) > diff_threshold:
            label = f"{label}_AI99"

        rows.append({
            "filename": os.path.basename(filepath),
            "top_species": str(label).strip(),
            "confidence": prob1  # <-- keep top-1 probability
        })

    if not rows:
        print("       ⚠️ No valid predictions to save.")
        return

    # Write CSV next to the JSON with *_topspecies.csv name
    output_path = json_path.with_name(
        json_path.name.replace("_predictions_relative.json", "_topspecies.csv")
    )

    # Ensure column order + consistent float formatting
    df = pd.DataFrame(rows, columns=["filename", "top_species", "confidence"])
    df.to_csv(output_path, index=False, float_format=f"%.{decimals}f")
    print("       ✅ Done")


def check_preview_class_distribution(preview_dir, threshold=0.3):
    print("⚙ Running preview classification check...")

    preview_tifs = list(preview_dir.glob("*_preview.tif"))
    if not preview_tifs:
        print("       ⚠️ No preview images found.")
        return True, None  # Allow pipeline to continue

    pred_lab, pred_prob = predict(model, preview_tifs, conf, top_K=1, filemode='local')
    top1_classes = [class_names[idx] for idx in pred_lab[:, 0]]

    class_counts = pd.Series(top1_classes).value_counts(normalize=True)
    print(f"       ✅ Class distribution in preview: {class_counts.to_dict()}")

    bubble_classes = [cls for cls in class_counts.index if 'bubbles' in cls.lower()]
    bubble_fraction = class_counts[bubble_classes].sum()

    if bubble_fraction > threshold:
        print(f"✅ Combined 'bubbles'-like classes exceed threshold ({threshold:.0%}): {bubble_fraction:.2%}, moved to quarantine")
        return False, 'bubbles'

    return True, None

def extract_only_morphology(tar_file, extract_dir, csv_path):
    base_name = tar_file.stem
    FILEPATHS = list(extract_dir.rglob("*.tif"))
    FILEPATHS = [p for p in FILEPATHS if "Background.tif" not in p.name and "FlowCellEdges.tif" not in p.name]

    if not FILEPATHS:
        print(f"       ⚠️ No valid .tif files in {base_name}, skipping morphology.")
        return

    print(f"       🧬 Extracting morphology for {len(FILEPATHS)} TIFFs...")

    results_csv = []

    for path in FILEPATHS:
        rel_path = str(path.relative_to(extract_dir))
        try:
            props = getMaxAreaDict(path)
            props["filepath"] = rel_path
            results_csv.append(props)
        except Exception as e:
            print(f"       ❌ Morphology error for {rel_path}: {e}")

    if results_csv:
        pd.DataFrame(results_csv).to_csv(csv_path, index=False)
        print(f"       ✅ Saved image properties CSV: {csv_path.name}")
    else:
        print(f"       ⚠️ No morphology data written for {base_name}")


def safe_bg_coords(bg_path):
    lat, lon = get_background_coordinates(bg_path)
    return (lat if lat is not None else 0, lon if lon is not None else 0)


def log_per_minute_metrics(tar_name, json_output, hits_file, exif_df, out_dir, num_images):
    import os
    print(f"⚙ Starting per-minute bio metrics")

    # Volume per minute (fixed)
    V_m3 = 0.034  # 34 L/min = 0.034 m³/min

    try:
        # --- 0) Load hits/misses ---
        if not hits_file.exists():
            print(f"⚠️ hitsmisses.txt missing for {tar_name}")
            return

        try:
            df_hits = pd.read_csv(hits_file)  # with header
        except Exception:
            df_hits = pd.read_csv(hits_file, header=None)
            df_hits.columns = ["hits", "misses"]

        # --- 1) Directly assign tar_name to tar_source column ---
        print(f"Assigning tar_source: {tar_name}")  # Check tar_name here
        df_hits["tar_source"] = tar_name  # Directly assign the tar_name to tar_source column
        print(f"Assigned tar_source: {df_hits['tar_source'].head()}")

        if "minute" not in df_hits.columns:
            df_hits["minute"] = range(len(df_hits))

        # Ensure necessary columns exist
        if "GPSLatitude" not in df_hits.columns:
            df_hits["GPSLatitude"] = "NA"  # Initialize as NA if missing
        if "GPSLongitude" not in df_hits.columns:
            df_hits["GPSLongitude"] = "NA"  # Initialize as NA if missing
        if "total_images_in_tar" not in df_hits.columns:
            df_hits["total_images_in_tar"] = num_images  # You can set this directly from the input argument

        max_minute = int(df_hits["minute"].max())

        # --- 2) Prepare EXIF → assign minutes sequentially ---
        lat_by_minute, lon_by_minute = {}, {}
        if exif_df is not None and not exif_df.empty:
            df_exif = exif_df.copy()

            ts_cols = [c for c in ["DateTimeOriginal", "FileModifyDate", "ModifyDate"] if c in df_exif.columns]

            if ts_cols:
                df_exif["capture_dt"] = pd.NaT
                for col in ts_cols:
                    s = pd.to_datetime(
                        df_exif[col],
                        format="%Y:%m:%d %H:%M:%S",  # EXIF datetime format
                        errors="coerce",
                        utc=True
                    )
                    # make tz-naive so it matches df_exif["capture_dt"]
                    if s.dt.tz is not None:
                        s = s.dt.tz_localize(None)

                    mask = df_exif["capture_dt"].isna()
                    df_exif.loc[mask, "capture_dt"] = s[mask]

                df_exif = df_exif.sort_values("capture_dt").reset_index(drop=True)

            else:
                df_exif = df_exif.reset_index(drop=True)

            n = len(df_exif)
            rows_per_min = max(1, n // (max_minute + 1))
            df_exif["minute"] = df_exif.index // rows_per_min
            df_exif["minute"] = df_exif["minute"].clip(0, max_minute)

            for c in ["GPSLatitude", "GPSLongitude"]:
                if c in df_exif.columns:
                    df_exif[c] = pd.to_numeric(df_exif[c], errors="coerce")

            coords_df = (df_exif
                         .dropna(subset=["GPSLatitude", "GPSLongitude"])
                         .groupby("minute", as_index=False)[["GPSLatitude", "GPSLongitude"]]
                         .median())
            lat_by_minute = dict(zip(coords_df["minute"], coords_df["GPSLatitude"]))
            lon_by_minute = dict(zip(coords_df["minute"], coords_df["GPSLongitude"]))

            exif_df = df_exif

        # --- 3) Map EXIF data to df_hits for GPS and fill in missing values ---
        # Assign GPS values from EXIF if they exist
        df_hits["GPSLatitude"] = df_hits["minute"].map(lambda m: lat_by_minute.get(m, "NA"))
        df_hits["GPSLongitude"] = df_hits["minute"].map(lambda m: lon_by_minute.get(m, "NA"))

        # --- 4) noctiluca + bubble counts per minute ---
        noct_counts = {m: 0 for m in df_hits["minute"]}
        bubble_counts = {m: 0 for m in df_hits["minute"]}

        if json_output.exists() and exif_df is not None and "tif_name" in exif_df.columns:
            name_to_minute = exif_df.set_index("tif_name")["minute"].to_dict()
            with open(json_output, "r") as f:
                data = json.load(f)

            for entry in data:
                fname = os.path.basename(entry.get("filepath", ""))
                labels = entry.get("top3_labels", [])
                if isinstance(labels, str):
                    labels = [s.strip() for s in labels.split(",")]

                m = name_to_minute.get(fname, None)
                if m is not None and len(labels) >= 1:
                    if "noct" in labels[0].lower():
                        noct_counts[m] = noct_counts.get(m, 0) + 1
                    if any("bubb" in lab.lower() for lab in labels):
                        bubble_counts[m] = bubble_counts.get(m, 0) + 1

        # --- 5) Morphometrics ---
        diameter_mean = {m: 0 for m in df_hits["minute"]}
        diameter_sum = {m: 0 for m in df_hits["minute"]}

        img_props_path = out_dir / f"{tar_name}_image_properties.csv"
        if img_props_path.exists() and exif_df is not None and "tif_name" in exif_df.columns:
            df_props = pd.read_csv(img_props_path)
            if "filepath" in df_props.columns and "object_additional_diameter_equivalent" in df_props.columns:
                # build mapping: filename -> minute
                name_to_minute = exif_df.set_index("tif_name")["minute"].to_dict()

                for _, row in df_props.iterrows():
                    fname = os.path.basename(str(row["filepath"]))
                    m = name_to_minute.get(fname, None)
                    if m is not None:
                        d = row["object_additional_diameter_equivalent"]
                        if pd.notna(d):
                            diameter_mean[m] = (diameter_mean.get(m, 0) + d)
                            diameter_sum[m] = (diameter_sum.get(m, 0) + d)

                # Turn sums into means (divide by noctiluca_count where >0)
                for m in diameter_mean:
                    if noct_counts.get(m, 0) > 0:
                        diameter_mean[m] = diameter_mean[m] / noct_counts[m]
                    else:
                        diameter_mean[m] = 0

        # attach to df_hits
        df_hits["noctiluca_diam_mean"] = df_hits["minute"].map(diameter_mean)
        df_hits["noctiluca_diam_sum"] = df_hits["minute"].map(diameter_sum)

        # --- 6) Merge and compute densities ---
        df_hits["noctiluca_count"] = df_hits["minute"].map(noct_counts)
        df_hits["bubbles"] = df_hits["minute"].map(bubble_counts)

        # Proportion of Noctiluca in Hits (for density)
        proportion_noctiluca_in_hits = df_hits["noctiluca_count"] / df_hits["hits"]

        # Calculate Noctiluca in Misses (based on the proportion in hits)
        df_hits["noctiluca_in_misses"] = proportion_noctiluca_in_hits * df_hits["misses"]

        # Total noctiluca count (hits + misses)
        df_hits["total_noctiluca"] = df_hits["noctiluca_count"] + df_hits["noctiluca_in_misses"]

        # Densities (individuals per m³)
        df_hits["noctiluca_density_ind_m3"] = df_hits["total_noctiluca"] / V_m3

        # Reorder columns
        df_hits = df_hits[[
            "tar_source", "minute", "hits", "misses",
            "bubbles", "noctiluca_count", "total_noctiluca", "noctiluca_density_ind_m3",
            "GPSLatitude", "GPSLongitude", "total_images_in_tar", "noctiluca_diam_mean", "noctiluca_diam_sum"
        ]]

        # Force proper dtypes before saving
        numeric_cols = [
            "minute", "hits", "misses",
            "bubbles","noctiluca_count", "total_noctiluca", "noctiluca_density_ind_m3",
            "GPSLatitude", "GPSLongitude", "total_images_in_tar", "noctiluca_diam_mean", "noctiluca_diam_sum"
        ]

        for col in numeric_cols:
            if col in df_hits.columns:
                df_hits[col] = pd.to_numeric(df_hits[col], errors="coerce")

        # Save clean CSV
        print(f"Final columns before saving CSV: {df_hits.columns}")
        out_path = out_dir / f"{tar_name}_bio-metrics.csv"
        df_hits.to_csv(out_path, index=False, float_format="%.6f")  # control decimals
        print(f"✅ Saved per-minute bio metrics: {out_path.name}")

    except Exception as e:
        print(f"❌ Failed per-minute log for {tar_name}: {e}")
        try:
            out_path = out_dir / f"{tar_name}_bio-metrics.csv"
            df_hits.to_csv(out_path, index=False)
        except Exception:
            pass



def map_exif_to_minutes(exif_df, hits_len):
    # Example: use the file index pattern from filename "_0001_"
    exif_df["minute"] = None
    for i, row in exif_df.iterrows():
        fname = row.get("tif_name", "")
        for m in range(hits_len):
            if f"_{m:04d}_" in fname or f"_{m:03d}_" in fname:
                exif_df.at[i, "minute"] = m
                break
    return exif_df
def clean_coord(value):
    # If value is tuple or list, flatten to string
    if isinstance(value, (tuple, list)):
        return ",".join(map(str, value))
    return value if value is not None else "NA"


# === MAIN PROCESS ===
def process_tar(tar_file):
    tar_name = tar_file.stem
    print(f"\n🔧🔧🔧 PROCESSING {tar_name.upper()} 🔧🔧🔧")

    # Tracking variables
    times_dict = {}
    num_images = 0
    status_log = []
    quarantined = False
    quarantine_reason = None
    already_logged = False
    pred_lab = None
    exif_df = None  # to pass into log_per_minute_metrics later

    if outputs_exist_for_tar(tar_file):
        print(f"📦 All outputs exist for {tar_name}, skipping.")
        return

    # Define paths
    json_output = source_dir / f"{tar_name}_predictions_relative.json"
    csv_output = source_dir / f"{tar_name}_image_properties.csv"
    topspecies_csv = source_dir / f"{tar_name}_topspecies.csv"
    exif_csv_path = source_dir / f"{tar_name}_gpstag.csv"
    hits_path = source_dir / f"{tar_name}_hitsmisses.txt"
    bg_path = source_dir / f"{tar_name}_Background.tif"
    preview_dir = Path(r"/paths/plankton-imager-10/not_processed/previews")
    preview_output = preview_dir / f"{tar_name}_preview"
    tar_dest = paths['tarred'] / tar_file.name
    extract_dir = paths['untarred'] / tar_name

    try:
        # === Step 1: Copy TAR to work dir ===
        start_time = time.time()
        shutil.copy(tar_file, tar_dest)
        status_log.append("TAR copied to working directory")
        times_dict["Copy TAR to working directory"] = track_time(start_time, "Copy TAR to working directory")

        # === Step 2: Untar ===
        start_time = time.time()
        clear_untarred_dir(paths['untarred'])
        extract_dir.mkdir()
        extract_tar(tar_dest, extract_dir)
        status_log.append("Untarred successfully")
        times_dict["Untar"] = track_time(start_time, "Untar")

        # === Step 3: Extract hitsmisses.txt ===
        start_time = time.time()
        if not hits_path.exists():
            ok = extract_hitsmisses(tar_file, hits_path, tar_file, status_log)
            if not ok:
                quarantined = True
                quarantine_reason = "hits/misses row count != 10"
                status_log.append(f"Quarantined: {quarantine_reason}")
                quarantine_target = quarantine_hitsmiss_dir / tar_file.name
                try:
                    shutil.move(str(tar_file), str(quarantine_target))
                except Exception as mv_err:
                    status_log.append(f"⚠️ Quarantine move failed: {mv_err}")
        else:
            status_log.append("hitsmisses.txt already exists (skipped)")
        times_dict["Extract hitsmisses.txt"] = track_time(start_time, "Extract hitsmisses.txt")

        # === Step 4–11 only if not quarantined ===
        if not quarantined:
            # Step 4: Count images
            start_time = time.time()
            num_images = count_images_in_tar(extract_dir, tar_file)
            status_log.append(f"Number of images in TAR: {num_images}")
            times_dict["Count images"] = track_time(start_time, "Count images")

            # Step 5: Preview images
            start_time = time.time()
            if not preview_output.exists():
                preview_dir.mkdir(parents=True, exist_ok=True)
                create_preview_images(extract_dir, preview_dir, tar_name)
                status_log.append("Preview images created")
            else:
                status_log.append("Preview images already exist (skipped)")
            times_dict["Create preview images"] = track_time(start_time, "Create preview images")

            # Step 6: Early preview classification
            start_time = time.time()
            if json_output.exists():
                print("✅ Skipping preview classification (predictions already exist)")
                should_continue, reason = True, None
            else:
                should_continue, reason = check_preview_class_distribution(preview_output, threshold=0.9)
            status_log.append(f"Preview classification result: {reason if reason else 'OK'}")
            times_dict["Early preview classification"] = track_time(start_time, "Early preview classification")

            if not should_continue:
                quarantined = True
                quarantine_reason = reason
                quarantine_target = quarantine_bubbles_dir / tar_file.name
                try:
                    shutil.move(str(tar_file), str(quarantine_target))
                    print(f"🚨 Quarantined {tar_file.name} → bubble issue")
                except Exception as mv_err:
                    status_log.append(f"⚠️ Quarantine move failed: {mv_err}")
                status_log.append(f"Moved to quarantine due to '{quarantine_reason}' class")

            # Step 7–11 only if still not quarantined
            if not quarantined:
                # Step 7: Copy Background.tif
                start_time = time.time()
                if not bg_path.exists():
                    copy_background_tif(extract_dir, bg_path)
                    if bg_path.exists():
                        status_log.append("Background.tif copied successfully")
                    else:
                        status_log.append("❌ Background.tif missing after copy attempt")
                else:
                    status_log.append("Background.tif already exists (skipped)")
                times_dict["Copy Background.tif"] = track_time(start_time, "Copy Background.tif")

                # Step 8: Extract EXIF metadata (or load existing)
                start_time = time.time()
                if not exif_csv_path.exists():
                    tif_files = list(extract_dir.rglob("*.tif"))
                    exif_df = extract_exif_metadata(tif_files, tar_name)
                    write_exif_csvs(exif_df, tar_name, paths['output'], source_dir)
                    status_log.append("EXIF metadata extracted and saved")
                else:
                    try:
                        exif_df = pd.read_csv(exif_csv_path)
                        status_log.append("EXIF metadata loaded from CSV")
                    except Exception as rd_err:
                        status_log.append(f"⚠️ Failed to load existing EXIF CSV: {rd_err}")
                        exif_df = None
                    print("✅ Skipping EXIF extraction (already exists)")

                    # Debug print for GPS check
                    #if exif_df is not None and not exif_df.empty:
                    #    print("🔎 EXIF DataFrame head:")
                    #    print(exif_df.head(5).to_string())
                    #else:
                    #    print("⚠️ EXIF DataFrame is empty or missing columns")

                times_dict["Extract and save EXIF metadata"] = track_time(start_time, "Extract and save EXIF metadata")

                # Step 9: Classification & morphology
                start_time = time.time()
                if not csv_output.exists():
                    if json_output.exists():
                        extract_only_morphology(tar_file, extract_dir, csv_output)
                        status_log.append("Image properties CSV created from existing predictions")
                    else:
                        pred_lab = classify_and_extract_regions(tar_file, extract_dir)
                        status_log.append("Classification and morphology run together (both files missing)")
                elif not json_output.exists():
                    pred_lab = classify_and_extract_regions(tar_file, extract_dir)
                    status_log.append("Re-ran full classification due to missing JSON")
                else:
                    status_log.append("Classification already exists (skipped)")
                times_dict["Classification and morphology extraction"] = track_time(
                    start_time, "Classification and morphology extraction"
                )

                # Step 10: Generate top species CSV
                start_time = time.time()
                if not topspecies_csv.exists():
                    if json_output.exists():
                        generate_topspecies_csv(json_output)
                        status_log.append("Top species CSV generated")
                    else:
                        status_log.append("Top species CSV skipped (JSON not found)")
                else:
                    status_log.append("Top species CSV already exists (skipped)")
                times_dict["Generate top species CSV"] = track_time(
                    start_time, "Generate top species CSV"
                )

                # Step 11: Per-minute bio metrics
                start_time = time.time()
                try:
                    #print(f"⚙ Starting per-minute bio metrics")
                    log_per_minute_metrics(
                        tar_name,
                        json_output,
                        hits_path,
                        exif_df,
                        source_dir,
                        num_images
                    )

                    status_log.append("Per-minute bio metrics logged")
                except Exception as e:
                    print(f"❌ Failed per-minute log for {tar_name}: {e}")
                    status_log.append(f"❌ Failed per-minute bio metrics: {e}")
                times_dict["Per-minute bio metrics"] = track_time(start_time, "Per-minute bio metrics")

    except Exception as e:
        status_log.append(f"❌ Unexpected error: {e}")

    finally:
        # Cleanup tar + untar
        try:
            if tar_dest.exists():
                os.remove(tar_dest)
            clear_untarred_dir(paths['untarred'])
        except Exception as cleanup_err:
            status_log.append(f"⚠️ Cleanup failed: {cleanup_err}")

        # Daily report
        daily_tar_reports.append({
            "tar_name": tar_name,
            "status_log": status_log,
            "quarantine_reason": quarantine_reason if quarantined else None
        })

        # Times log
        if not already_logged:
            try:
                log_time_to_file(tar_name, times_dict, num_images)
            except Exception as log_err:
                print(f"⚠️ Failed to log times for {tar_name}: {log_err}")

        print(f"🔧🔧🔧 DONE {tar_name} 🔧🔧🔧")

        # Mark done
        try:
            done_marker = source_dir / f"{tar_name}.done"
            with open(done_marker, "w") as f:
                f.write(f"Processed at {datetime.datetime.now()}\n")
            #print(f"✅ Wrote done marker for {tar_name}")
        except Exception as e:
            print(f"⚠️ Could not write done marker for {tar_name}: {e}")


# === CONTINUOUS WATCH ===
print("⚙  Watching for new .tar files (press Ctrl+C to stop)...")
while True:
    now = datetime.datetime.now()
    today_str = now.strftime('%Y-%m-%d')

    new_files = get_new_tar_files(source_dir)
    pipeline_count = len(new_files)

    if pipeline_count == 0:
        print(f"[{time.ctime()}] No new .tar files. Sleeping...")
        time.sleep(3600)
    else:
        print(f"[{time.ctime()}] ✅ {pipeline_count} .tar file(s) in the processing pipeline.")
        for tar_file in new_files:
            lockfile = source_dir / f"{tar_file.stem}.lock"
            if lockfile.exists():
                continue
            try:
                lockfile.touch(exist_ok=False)
                process_tar(tar_file)
            except Exception as e:
                print(f"❌ Failed to process {tar_file.name}: {e}")
            finally:
                if lockfile.exists():
                    lockfile.unlink()
        print("🔁 Rechecking in 1 hour...")
