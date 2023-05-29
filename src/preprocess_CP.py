
import os
import shutil
import csv
import glob
import numpy as np
import pandas as pd
import h5py
import nibabel as nib
import scipy.ndimage
from datetime import datetime
import random
# import defaultdict
import pdb

import os

def subj_to_scan_id():
    p_file = "PatientDict.txt"
    mapping = {} #defaultdict(list)
    # associates list of scanID (value) to key (patient number)
    
    with open(p_file, 'r') as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace and newline characters
            key, value = line.split('\t')
            if key in mapping:
                mapping[key].append(value)
            else:
                mapping[key] = [value]
    
    # new_mapping = {}

    # counter = 1
    # for key, values in mapping.items():
    #     new_mapping[counter] = values
    #     counter += 1

    return mapping

def create_participant_file(output_dir):
    mapping = subj_to_scan_id()
    filename = 'participants.tsv'

    xl_path = '/media/andjela/SeagatePor1/CP/Calgary_Preschool_Dataset_Updated_20200213_copy.xlsx' 
    df = pd.read_excel(xl_path)

    with open(f'{output_dir}/{filename}', "w") as file:
        writer = csv.writer(file, delimiter='\t')
    
        # Write header row
        writer.writerow(['participant_id', 'scan_id', 'session', 'age', 'sex', 'group'])
        group = 'control'
        
        # Write data rows
        count_sub = 0
        count_ses = 0
        for sub, scan_ids in mapping.items():
            for scan_id in scan_ids:
                age = df.loc[df['ScanID'] == scan_id, 'Age (Years)'].values[0]
                sex = df.loc[df['ScanID'] == scan_id, 'Biological Sex (Female = 0; Male = 1)'].values[0]
                writer.writerow(['sub-{:03d}'.format(count_sub+1), f'{scan_id}', 'ses-{:03d}'.format(count_ses+1), f'{age}', f'{sex}', f'{group}' ])
                count_ses += 1
            count_ses = 0
            count_sub += 1

def convert_to_bids(input_dir, output_dir):
    # Get a list of all folders in the input directory
    folders = [folder for folder in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, folder))]

    mapping = subj_to_scan_id()
    # Iterate over each folder and convert the names
    for folder in folders:
        subj_id_scan_id = folder.split("_")  # Split subjID and scanID using "_"
        subj_id = subj_id_scan_id[0]  # Extract the subj_id (e.g., '10006')
        scan_id = "_".join(subj_id_scan_id[1:])  # Extract the scan_id (e.g., 'PS14_001')

        for key, values in mapping.items():
            if scan_id in values:
                scan_id_idx = values.index(scan_id) +1

        subj_id_idx = list(mapping).index(f'{subj_id}')+1
        
        new_subj_id = "sub-{:03d}".format(subj_id_idx) # Convert subjID to sub_001, sub_002, etc.
        new_scan_id = "ses-{:03d}".format(scan_id_idx)  # Convert scanID to ses-001, ses-002, etc.
        
        # Create the new BIDS folder structure
        new_folder = os.path.join(output_dir, new_subj_id, new_scan_id, 'anat')
        os.makedirs(new_folder, exist_ok=True)

        # Move the contents of the original folder to the new BIDS folder
        original_folder = os.path.join(input_dir, folder)
        for file in os.listdir(original_folder):
            if file.endswith("dtype.nii.gz"):
                file_path = os.path.join(original_folder, file)
                # session_number = int(scan_id)
                new_file_name = "sub-{:03d}_ses-{:03d}_rigid_T1w.nii.gz".format(subj_id_idx, scan_id_idx)
                new_file_path = os.path.join(new_folder, new_file_name)
                shutil.copy(file_path, new_file_path)
        
        print("Converted {} to {}".format(folder, new_folder))


if __name__ == "__main__":
    input_directory = "/media/andjela/SeagatePor1/CP/rigid/"
    output_directory = "/media/andjela/SeagatePor1/LSSL/data/"

    # convert_to_bids(input_directory, output_directory)

    create_participant_file(output_directory)