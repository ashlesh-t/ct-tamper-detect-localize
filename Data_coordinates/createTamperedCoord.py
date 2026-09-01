import argparse
import os
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
import pydicom
import numpy as np
from collections import defaultdict

parser = argparse.ArgumentParser(description="Extract tumor centroid coordinates from LIDC-style DICOM+XML annotations.")
parser.add_argument("--data-root", required=True, help="Path to the dataset folder containing UUID subfolders with DICOMs + XML annotations")
parser.add_argument("--output-csv", default="centroids_cancer_unique.csv", help="Path to write the output CSV")
args = parser.parse_args()

# Main dataset folder (UUID folders with DICOMs + XML)
main_folder = Path(args.data_root)

# Namespace in XML files
namespace = {'ns': 'http://www.nih.gov'}

# Output CSV
csv_filename = Path(args.output_csv).expanduser()
os.makedirs(csv_filename.parent, exist_ok=True)

# Dictionary: key = uuid_folder, value = dict(slice_number -> list of (x, y))
tumor_slices = defaultdict(lambda: defaultdict(list))

def calculate_centroid_for_roi(roi):
    x_coords, y_coords = [], []
    for edge in roi.findall("ns:edgeMap", namespace):
        x_elem = edge.find("ns:xCoord", namespace)
        y_elem = edge.find("ns:yCoord", namespace)
        if x_elem is not None and y_elem is not None:
            try:
                x_coords.append(float(x_elem.text))
                y_coords.append(float(y_elem.text))
            except (TypeError, ValueError):
                continue
    if x_coords and y_coords:
        return sum(x_coords)/len(x_coords), sum(y_coords)/len(y_coords)
    return None, None

def get_dicom_slices(uuid_folder_path):
    dicom_files = [f for f in os.listdir(uuid_folder_path) if f.lower().endswith(".dcm")]
    slices = []
    for f in dicom_files:
        try:
            ds = pydicom.dcmread(os.path.join(uuid_folder_path, f), stop_before_pixels=True)
            z = float(ds.ImagePositionPatient[2])
            instance = int(ds.InstanceNumber)
            slices.append((z, instance))
        except Exception:
            continue
    slices.sort(key=lambda x: x[0])
    return slices

def find_closest_slice(z_value, dicom_slices):
    z_array = np.array([s[0] for s in dicom_slices])
    idx = (np.abs(z_array - z_value)).argmin()
    return dicom_slices[idx][1]

def extract_data_from_xml(xml_path, uuid_folder_path):
    try:
        dicom_slices = get_dicom_slices(uuid_folder_path)
        if not dicom_slices:
            return

        tree = ET.parse(xml_path)
        root = tree.getroot()
        unblinded_nodules = root.findall(".//ns:unblindedReadNodule", namespace)

        for unblinded_nodule in unblinded_nodules:
            # Check malignancy
            characteristics = unblinded_nodule.find("ns:characteristics", namespace)
            if characteristics is None:
                continue
            malignancy_elem = characteristics.find("ns:malignancy", namespace)
            if malignancy_elem is None or not malignancy_elem.text:
                continue
            try:
                malignancy = int(malignancy_elem.text)  
            except ValueError:
                continue
            if malignancy >= 4:
                continue

            rois = unblinded_nodule.findall("ns:roi", namespace)
            for roi in rois:
                centroid_x, centroid_y = calculate_centroid_for_roi(roi)
                if centroid_x is None or centroid_y is None:
                    continue
                z_elem = roi.find("ns:imageZposition", namespace)
                if z_elem is None or not z_elem.text:
                    continue
                try:
                    roi_z = float(z_elem.text)
                except ValueError:
                    continue

                slice_number = find_closest_slice(roi_z, dicom_slices)
                tumor_slices[uuid_folder_path.name][slice_number].append((centroid_x, centroid_y))

    except Exception as e:
        print(f"Error processing XML {xml_path}: {e}")

# Walk dataset
for root_dir, dirs, files in os.walk(main_folder):
    for file in files:
        if file.lower().endswith(".xml"):
            uuid_folder_path = Path(root_dir)
            xml_file_path = os.path.join(root_dir, file)
            extract_data_from_xml(xml_file_path, uuid_folder_path)

# Prepare final data: one row per slice, average x,y for duplicates
final_rows = []
for uuid, slices_dict in tumor_slices.items():
    for slice_num in sorted(slices_dict.keys()):
        points = slices_dict[slice_num]
        avg_x = sum(p[0] for p in points)/len(points)
        avg_y = sum(p[1] for p in points)/len(points)
        final_rows.append((uuid, avg_x, avg_y, slice_num))

# Save CSV
df = pd.DataFrame(final_rows, columns=["filename", "x", "y", "slice_number"])
df.to_csv(csv_filename, index=False)

print(f"\nCSV saved to: {csv_filename}")
print(f"Total rows: {len(df)}")
