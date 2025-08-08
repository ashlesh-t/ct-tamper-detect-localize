import os
import xml.etree.ElementTree as ET
import pandas as pd
import pydicom as pdy

# Define the main folder
main_folder = r"/media/a/A/C/DataSet/LungsCT-BigData/manifest-1600709154662/LIDC-IDRI"

# XML namespace
namespace = {'ns': 'http://www.nih.gov'}

# Output CSV file
csv_filename = " ~/capstone/DataStore/centroids_cancer.csv"

# Use a set to store unique data
csv_data = set()

# Function to calculate centroid
def calculate_centroid(roi):
    x_coords = []
    y_coords = []
    for edge in roi.findall("ns:edgeMap", namespace):
        x_coord = edge.find("ns:xCoord", namespace)
        y_coord = edge.find("ns:yCoord", namespace)
        if x_coord is not None and y_coord is not None:
            x_coords.append(float(x_coord.text))
            y_coords.append(float(y_coord.text))
    if x_coords and y_coords:
        return sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords)
    return None, None

# # Function to get Z-index from DICOM files
# def get_zindex(dicom_folder, image_z):
#     image_z = float(image_z)  # Convert image_z to float for comparison
#     try:
#         for file in os.listdir(dicom_folder):
#             if file.endswith('.dcm'):
#                 dicom_path = os.path.join(dicom_folder, file)
#                 dy = pdy.dcmread(dicom_path)
#                 slice_location = dy.get("SliceLocation", None)
#                 if slice_location is not None and round(slice_location, 4) == round(image_z, 4):
#                     # Extract slice number from filename if available
#                     return int(file.split('.')[0].split('-')[-1])
#     except Exception as e:
#         print(f"Error reading DICOM in {dicom_folder}: {str(e)}")
#     return None

# Function to extract data from XML
def extract_data_from_xml(files, xml_path, series_folder):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        unblinded_nodule = root.find(".//ns:unblindedReadNodule", namespace)
        if unblinded_nodule is not None:
            for roi in unblinded_nodule.findall("ns:roi", namespace):
                image_z_elem = roi.find("ns:imageZposition", namespace)
                if image_z_elem is None:
                    continue

                image_z = image_z_elem.text
                centroid_x, centroid_y = calculate_centroid(roi)

                if centroid_x is not None and centroid_y is not None:
                    # z_index = get_zindex(os.path.dirname(xml_path), image_z)
                    # if z_index is not None:
                        print(series_folder)
                        csv_data.add((series_folder, centroid_x, centroid_y, image_z))

    except Exception as e:
        print(f"Error processing XML {xml_path}: {e}")

# Traverse the main folder recursively
for root, dirs, files in os.walk(main_folder):

    for file in files:
        if file.endswith(".xml"):
            parts = os.path.normpath(root).split(os.sep)[-3]
            # print(root)
            parent_folder = os.path.basename(os.path.dirname(root))

            # print(parent_folder)
            series_folder = os.path.basename(root)  # Get the folder containing XML
            xml_file_path = os.path.join(root, file)
            extract_data_from_xml(files, xml_file_path, parts +'/'+parent_folder+'/'+series_folder)

# Convert set to DataFrame and save to CSV
df = pd.DataFrame(sorted(csv_data), columns=["filename", "x", "y", "z"])
df.to_csv(csv_filename, index=False)

print(f"CSV file '{csv_filename}' created successfully with {len(df)} unique records!")
