import pandas as pd
import shutil
import os

# Load the CSV file
df = pd.read_csv('LET.csv')

# Filter the dataframe
df_filtered = df[df['quality'] == 1]

# Define the directories
dir1 = 'DR/'
dir2 = 'NO_DR/'
id1 = os.listdir('/home/sauravb/Tensorflow/train_model/sr_datasets/Mild/')
id2 = os.listdir('/home/sauravb/Tensorflow/train_model/sr_datasets/Moderate/')
id3 = os.listdir('/home/sauravb/Tensorflow/train_model/sr_datasets/No_DR/')
id4 = os.listdir('/home/sauravb/Tensorflow/train_model/sr_datasets/Proliferate_DR/')
id5 = os.listdir('/home/sauravb/Tensorflow/train_model/sr_datasets/Severe/')

# Ensure the output directories exist
os.makedirs(dir1, exist_ok=True)
os.makedirs(dir2, exist_ok=True)

# Combine the lists
image_list = id1 + id2 + id3 + id4 + id5

def dr_ch(src_dir,image_name):
    if dr_grade > 0:
        shutil.copy(src_dir + image_name, dir1)
    else:
        shutil.copy(src_dir + image_name, dir2)

# Iterate over the dataframe
for index, row in df_filtered.iterrows():
    image_name = row['image']
    dr_grade = row['DR_grade']
    # Check if image exists in the list
    if image_name in image_list:
        # Determine the source directory
        if image_name in id1:
            src_dir = '/home/sauravb/Tensorflow/train_model/sr_datasets/Mild/'
            dr_ch(src_dir, image_name)
        elif image_name in id2:
            src_dir = '/home/sauravb/Tensorflow/train_model/sr_datasets/Moderate/'
            dr_ch(src_dir, image_name)
        elif image_name in id3:
            src_dir = '/home/sauravb/Tensorflow/train_model/sr_datasets/No_DR/'
            dr_ch(src_dir, image_name)
        elif image_name in id4:
            src_dir = '/home/sauravb/Tensorflow/train_model/sr_datasets/Proliferate_DR/'
            dr_ch(src_dir, image_name)
        elif image_name in id5:
            src_dir = '/home/sauravb/Tensorflow/train_model/sr_datasets/Severe/'
            dr_ch(src_dir, image_name)
