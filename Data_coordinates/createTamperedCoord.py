import os
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path

# Main LIDC dataset folder
main_folder = r"/media/ashu/Ashlesh/CapstoneProject/DataSet/LungsCT-BigData/manifest-1600709154662/LIDC-IDRI"

# Namespace used in the XML files
namespace = {'ns': 'http://www.nih.gov'}

# Output CSV path
csv_filename = Path('centroids_cancer.csv').expanduser()
os.makedirs(csv_filename.parent, exist_ok=True)

# Dictionary: key = (series_folder, tumor_id), value = list of (x, y, z)
tumor_points = {}

def calculate_centroid_for_roi(roi):
    """Calculate centroid (x, y) for one ROI slice."""
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
        return sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords)
    return None, None

def extract_data_from_xml(xml_path, series_folder):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Find all nodules
        unblinded_nodules = root.findall(".//ns:unblindedReadNodule", namespace)
        for tumor_id, unblinded_nodule in enumerate(unblinded_nodules, start=1):

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
            if malignancy < 4:
                continue  # Skip benign/uncertain nodules

            # Collect all slice centroids for this tumor
            rois = unblinded_nodule.findall("ns:roi", namespace)
            for roi in rois:
                image_z_elem = roi.find("ns:imageZposition", namespace)
                if image_z_elem is None:
                    continue
                try:
                    image_z = float(image_z_elem.text)
                except ValueError:
                    continue

                centroid_x, centroid_y = calculate_centroid_for_roi(roi)
                if centroid_x is not None and centroid_y is not None:
                    tumor_points.setdefault((series_folder, tumor_id), []).append((centroid_x, centroid_y, image_z))

    except Exception as e:
        print(f"Error processing XML {xml_path}: {e}")

print(f"Scanning dataset under: {main_folder}")
for root, dirs, files in os.walk(main_folder):
    for file in files:
        if file.endswith(".xml"):
            parts = os.path.normpath(root).split(os.sep)[-3]
            parent_folder = os.path.basename(os.path.dirname(root))
            series_folder = os.path.basename(root)
            xml_file_path = os.path.join(root, file)
            extract_data_from_xml(xml_file_path, f"{parts}/{parent_folder}/{series_folder}")

# Compute one centroid per tumor
final_data = []
for (series_folder, _tumor_id), points in tumor_points.items():
    xs, ys, zs = zip(*points)
    centroid_x = sum(xs) / len(xs)
    centroid_y = sum(ys) / len(ys)
    centroid_z = sum(zs) / len(zs)
    final_data.append((series_folder, centroid_x, centroid_y, centroid_z))

# Save to CSV in the same format as before
df = pd.DataFrame(final_data, columns=["filename", "x", "y", "z"])
df.to_csv(csv_filename, index=False)

print(f"\nCSV file saved to: {csv_filename}")
print(f"Total malignant tumor centroids: {len(df)}")
