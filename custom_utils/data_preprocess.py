# Load packages
import pandas as pd
import os
import subprocess

# Preprocssing functions
def upzip_csv(root_dir):
    file_nms = ["chexpert.csv", "metadata.csv", "negbio.csv", "split.csv"]
    mimic_version = root_dir.split("/")[-2]
    for file_nm in file_nms:
        try:
            subprocess.run(["gunzip", os.path.join(root_dir, f"mimic-cxr-{mimic_version}-{file_nm}")])
        except:
            continue

def get_merged_csv(root_dir, labeling_method):
    upzip_csv(root_dir=root_dir)

    target_label_df = pd.read_csv(os.path.join(root_dir, f'mimic-cxr-2.0.0-{labeling_method}.csv'))
    metadata_df = pd.read_csv(os.path.join(root_dir, 'mimic-cxr-2.0.0-metadata.csv'))
    split_df = pd.read_csv(os.path.join(root_dir, 'mimic-cxr-2.0.0-split.csv'))

    return pd.merge(
        pd.merge(metadata_df, split_df, left_on=['dicom_id', 'study_id', 'subject_id'], right_on=['dicom_id', 'study_id', 'subject_id'], how='inner'),
        target_label_df, 
        left_on=['study_id', 'subject_id'], right_on=['study_id', 'subject_id'],
        how='right', # This is important due to the symmetric difference between metadata(or split) and mimic_chexpert
        )

def split_and_save(root_dir, merged_df, labeling_method):
    # Split
    train_df = merged_df[merged_df.split=='train']
    valid_df = merged_df[merged_df.split=='validate']
    test_df = merged_df[merged_df.split=='test']
    
    # Save
    train_df.to_csv(os.path.join(root_dir, f"train_{labeling_method}.csv"))
    valid_df.to_csv(os.path.join(root_dir, f"valid_{labeling_method}.csv"))
    test_df.to_csv(os.path.join(root_dir, f"test_{labeling_method}.csv"))

def get_splitted_mimic_csv(root_dir, labeling_method):
    split_and_save(
        root_dir=root_dir, 
        merged_df=get_merged_csv(
            root_dir=root_dir, 
            labeling_method=labeling_method
            ), 
        labeling_method=labeling_method,
        )


if __name__=="__main__":
    
    root_dir = '/home/mimic_pad224/2.0.0'

    get_splitted_mimic_csv(root_dir=root_dir, labeling_method='chexpert')
    get_splitted_mimic_csv(root_dir=root_dir, labeling_method='negbio')

    