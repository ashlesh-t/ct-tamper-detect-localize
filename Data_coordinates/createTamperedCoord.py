import os
import xml.etree.ElementTree as ET
import pandas as pd
import pydicom as pdy
from pathlib import Path

# Main LIDC dataset folder
main_folder = r"/media/ashu/Ashlesh/CapstoneProject/DataSet/LungsCT-BigData/manifest-1600709154662/LIDC-IDRI"

# Namespace used in the XML files
namespace = {'ns': 'http://www.nih.gov'}

# Output CSV path
csv_filename = Path('~/capstone/DataStore/centroids_cancer.csv').expanduser()
os.makedirs(csv_filename.parent, exist_ok=True)

# Set to store unique (filename, x, y, z)
csv_data = set()

# Function to calculate centroid of an ROI
def calculate_centroid(roi):
    x_coords = []
    y_coords = []
    edges = roi.findall("ns:edgeMap", namespace)
    for edge in edges:
        x_coord = edge.find("ns:xCoord", namespace)
        y_coord = edge.find("ns:yCoord", namespace)
        if x_coord is not None and y_coord is not None:
            try:
                x_coords.append(float(x_coord.text))
                y_coords.append(float(y_coord.text))
            except (TypeError, ValueError):
                continue
    if x_coords and y_coords:
        return sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords)
    return None, None


# Function to extract centroid & z from XML
def extract_data_from_xml(files, xml_path, series_folder):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Find all nodules
        unblinded_nodules = root.findall(".//ns:unblindedReadNodule", namespace)
        for unblinded_nodule in unblinded_nodules:

            # Only include nodules with malignancy >= 4
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
            if malignancy < 4:
                continue  # Skip benign/uncertain nodules

            rois = unblinded_nodule.findall("ns:roi", namespace)
            for roi in rois:
                image_z_elem = roi.find("ns:imageZposition", namespace)
                if image_z_elem is None:
                    continue

                image_z = image_z_elem.text
                centroid_x, centroid_y = calculate_centroid(roi)

                if centroid_x is not None and centroid_y is not None:
                    csv_data.add((series_folder, centroid_x, centroid_y, image_z))

    except Exception as e:
        print(f"Error processing XML {xml_path}: {e}")


# Traverse all subfolders and XMLs
print(f"Scanning dataset under: {main_folder}")
for root, dirs, files in os.walk(main_folder):
    for file in files:
        if file.endswith(".xml"):
            parts = os.path.normpath(root).split(os.sep)[-3]
            parent_folder = os.path.basename(os.path.dirname(root))
            series_folder = os.path.basename(root)
            xml_file_path = os.path.join(root, file)

            extract_data_from_xml(files, xml_file_path, f"{parts}/{parent_folder}/{series_folder}")

# Save to CSV
df = pd.DataFrame(sorted(csv_data), columns=["filename", "x", "y", "z"])
df.to_csv(csv_filename, index=False)

print(f"\nCSV file saved to: {csv_filename}")
print(f"Total malignant tumor centroids: {len(df)}")
