
import os
import glob
import numpy as np
import pandas as pd
import h5py
import nibabel as nib
import scipy.ndimage
from datetime import datetime
import random
import pdb

def create_subj_data(data_path, file_ending, tsv_path):
    # load label, age, image paths
    # File named following sub-001_ses-001_rigid_T1w_64.nii.gz
    '''
    struct subj_data

    age: baseline age,
    label: label for the subject, 0 - NC, 2 - AD, 3 - sMCI, 4 - pMCI
    label_all: list of labels for each timestep, 0 - NC, 1 - MCI, 2 - AD
    date_start: baseline date, in datetime format
    date: list of dates, in datetime format
    date_interval: list of intervals, in year
    img_paths: list of image paths
    '''
    df_raw = pd.read_csv(tsv_path, usecols=['participant_id', 'scan_id', 'session', 'age', 'sex', 'group'], sep='\t')

    img_paths = glob.glob(data_path+f'*/*/*/*{file_ending}')
    img_paths = sorted(img_paths)
    subj_data = {}
    # label_dict = {'C': 0, 'E': 1, 'H': 2, 'HE': 3} #why so many labels
    # nan_label_count = 0
    # nan_idx_list = []
    for img_path in img_paths:
        subj_id = os.path.basename(img_path).split('_')[0]
        ses_id = os.path.basename(img_path).split('_')[1]
        # date = os.path.basename(img_path).split('-')[1].split('.')[0]
        # # date = os.path.basename(img_path).split('-')[1] + '-' + os.path.basename(img_path).split('-')[2] + '-' + os.path.basename(img_path).split('-')[3].split('_')[0]
        
        # date_struct = datetime.strptime(date, '%Y%m%d')
        rows = df_raw.loc[(df_raw['participant_id'] == subj_id)] #nbr_sujetsx6
        # Age threshold to keep only children older than 3 years of age
        # rows = rows[rows['age']>3]
        # ses_id = rows.iloc[0]['session'] 

        age_init = rows.iloc[0]['age']
        sex = rows.iloc[0]['sex']
        age = rows.loc[rows['session'] == f'{ses_id}', 'age'].values[0]
        if rows.shape[0] == 0:
            print('Missing label for', subj_id)
        else:
            # build dict
            # label = rows.iloc[0]['demo_diag']
            # if label not in label_dict.keys(): # ignore other labels
            #     continue
            # Add this condition to keep only ages greater than 3
            if age > 3:  
                if subj_id not in subj_data:
                    # dob =  rows.iloc[0]['demo_dob']
                    # if dob == 'NaT':
                    #     pdb.set_trace()
                    # age = (date_struct - datetime.strptime(dob, '%Y-%m-%d')).days / 365.
                    subj_data[subj_id] = {'age': age_init, 'sex': sex, 'ages': [], 'age_interval': [], 'img_paths': []}
                # if age not in subj_data[subj_id]['ages']:
                subj_data[subj_id]['ages'].append(age)
                subj_data[subj_id]['age_interval'].append((age - subj_data[subj_id]['age']))
                subj_data[subj_id]['img_paths'].append(os.path.basename(img_path))
            else:
                print(subj_id)
    return subj_data


# # get sMCI, pMCI labels
# num_ts_c = 0
# num_ts_h = 0
# num_ts_e = 0
# num_ts_he = 0
# num_c = 0
# num_h = 0
# num_e = 0
# num_he = 0
# subj_list_dict = {'C':[], 'H':[], 'E': [], 'HE': []}
# for subj_id in subj_data.keys():
#     if subj_data[subj_id]['label'] == 0:
#         num_c += 1
#         num_ts_c += len(subj_data[subj_id]['img_paths'])
#         subj_list_dict['C'].append(subj_id)
#     elif subj_data[subj_id]['label'] == 1:
#         num_e += 1
#         num_ts_e += len(subj_data[subj_id]['img_paths'])
#         subj_list_dict['E'].append(subj_id)
#     elif subj_data[subj_id]['label'] == 2:
#         num_h += 1
#         num_ts_h += len(subj_data[subj_id]['img_paths'])
#         subj_list_dict['H'].append(subj_id)
#     else:
#         num_he += 1
#         num_ts_he += len(subj_data[subj_id]['img_paths'])
#         subj_list_dict['HE'].append(subj_id)

# print('Number of timesteps, C/E/H/HE:', num_ts_c, num_ts_e, num_ts_h, num_ts_he)
# print('Number of subject, C/E/H/HE:', num_c, num_e, num_h, num_he)

# # save subj_list_dict to npy
# np.save('/data/jiahong/data/LAB/LAB_longitudinal_subj.npy', subj_list_dict)

def stats_age(subj_data):
    # statistics about timesteps
    max_timestep = 0
    num_cls = [0,0,0,0,0]
    num_ts = np.zeros((15,))
    counts = np.zeros((4, 15))
    for subj_id, info in subj_data.items():
        num_timestep = len(info['img_paths'])
        max_timestep = max(max_timestep, num_timestep)
        num_cls[info['label']] += 1
        num_ts[num_timestep] += 1
        counts[info['label'], num_timestep] += 1
    print('Number of subjects: ', len(subj_data))
    print('Max number of timesteps: ', max_timestep)
    print('Number of each timestep', num_ts)
    print('Number of each class', num_cls)
    print('C', counts[0])
    print('E', counts[1])
    print('H', counts[2])
    print('HE', counts[3])

    counts_cum = counts.copy()
    for i in range(counts.shape[1]-2, 0, -1):
        counts_cum[:, i] += counts_cum[:, i+1]
    print(counts_cum)

# save subj_data to h5
def save_subj_data_h5(h5_noimg_path, subj_data):
    if not os.path.exists(h5_noimg_path):
        f_noimg = h5py.File(h5_noimg_path, 'a')
        for i, subj_id in enumerate(subj_data.keys()):
            subj_noimg = f_noimg.create_group(subj_id)
            # subj_noimg.create_dataset('label', data=subj_data[subj_id]['label'])
            # subj_noimg.create_dataset('label_all', data=subj_data[subj_id]['label_all'])

            # subj_noimg.create_dataset('age_interval', data=subj_data[subj_id]['age_interval'])
            subj_noimg.create_dataset('age', data=subj_data[subj_id]['age'])
            subj_noimg.create_dataset('sex', data=subj_data[subj_id]['sex'])
            subj_noimg.create_dataset('ages', data=subj_data[subj_id]['ages'])
            # subj_noimg.create_dataset('img_paths', data=subj_data[subj_id]['img_paths'])

# save images to h5
def save_imgs_h5(h5_img_path, subj_data):
    if not os.path.exists(h5_img_path):
        f_img = h5py.File(h5_img_path, 'a')
        for i, subj_id in enumerate(subj_data.keys()):
            subj_img = f_img.create_group(subj_id)
            img_paths = subj_data[subj_id]['img_paths']
            for img_path in img_paths:
                ses_id = os.path.basename(img_path).split('_')[1]
                
                img_nib = nib.load(os.path.join(data_path,subj_id, ses_id, 'anat', img_path))
                img = img_nib.get_fdata()
                # img = (img - np.mean(img)) / np.std(img) #already done in preprocess
                subj_img.create_dataset(os.path.basename(img_path), data=img)
            # print(i, subj_id)

def augment_image(img, rotate, shift, flip):
    # pdb.set_trace()
    img = scipy.ndimage.interpolation.rotate(img, rotate[0], axes=(1,0), reshape=False)
    img = scipy.ndimage.interpolation.rotate(img, rotate[1], axes=(0,2), reshape=False)
    img = scipy.ndimage.interpolation.rotate(img, rotate[2], axes=(1,2), reshape=False)
    img = scipy.ndimage.shift(img, shift[0])
    if flip[0] == 1:
        img = np.flip(img, 0) - np.zeros_like(img)
    return img

def generate_aug_data(h5_img_path, aug_size, subj_data):

    if not os.path.exists(h5_img_path):
        f_img = h5py.File(h5_img_path, 'a')
        for i, subj_id in enumerate(subj_data.keys()):
            subj_img = f_img.create_group(subj_id)
            img_paths = subj_data[subj_id]['img_paths']
            rotate_list = np.random.uniform(-2, 2, (aug_size-1, 3))
            shift_list =  np.random.uniform(-2, 2, (aug_size-1, 1))
            flip_list =  np.random.randint(0, 2, (aug_size-1, 1))
            for img_path in img_paths:
                ses_id = os.path.basename(img_path).split('_')[1]
                
                img_nib = nib.load(os.path.join(data_path,subj_id, ses_id, 'anat', img_path))
                img = img_nib.get_fdata()
                # img = (img - np.mean(img)) / np.std(img)
                imgs = [img]
                for j in range(aug_size-1):
                    imgs.append(augment_image(img, rotate_list[j], shift_list[j], flip_list[j]))
                imgs = np.stack(imgs, 0)
                subj_img.create_dataset(os.path.basename(img_path), data=imgs)
            print(i, subj_id)

def save_data_txt(path, subj_id_list, case_id_list):
    with open(path, 'w') as ft:
        for subj_id, case_id in zip(subj_id_list, case_id_list):
            ft.write(subj_id+' '+case_id+'\n')

def get_subj_case_id_list(subj_data, subj_id_list):
    subj_id_list_full = []
    case_id_list_full = []
    for subj_id in subj_id_list:
        case_id_list = subj_data[subj_id]['img_paths']
        case_id_list_full.extend(case_id_list)
        subj_id_list_full.extend([subj_id] * len(case_id_list))
    return subj_id_list_full, case_id_list_full

# save txt, subj_id, case_id, case_number, case_id, case_number
def save_pair_data_txt(path, subj_id_list, case_id_list):
    with open(path, 'w') as ft:
        for subj_id, case_id in zip(subj_id_list, case_id_list):
            ft.write(subj_id+' '+case_id[0]+' '+case_id[1]+' '+str(case_id[2])+' '+str(case_id[3])+'\n')

def save_single_data_txt(path, subj_id_list, case_id_list):
    with open(path, 'w') as ft:
        for subj_id, case_id in zip(subj_id_list, case_id_list):
            ft.write(subj_id+' '+case_id[0]+' '+str(case_id[1])+'\n')

def get_subj_pair_case_id_list(subj_data, subj_id_list):
    subj_id_list_full = []
    case_id_list_full = []
    for subj_id in subj_id_list:
        case_id_list = subj_data[subj_id]['img_paths']
        for i in range(len(case_id_list)):
            for j in range(i+1, len(case_id_list)):
                subj_id_list_full.append(subj_id)
                case_id_list_full.append([case_id_list[i],case_id_list[j],i,j])
                
    return subj_id_list_full, case_id_list_full

def get_subj_single_case_id_list(subj_data, subj_id_list):
    subj_id_list_full = []
    case_id_list_full = []
    for subj_id in subj_id_list:
        case_id_list = subj_data[subj_id]['img_paths']
        for i in range(len(case_id_list)):
            subj_id_list_full.append(subj_id)
            case_id_list_full.append([case_id_list[i], i])
    return subj_id_list_full, case_id_list_full

# pdb.set_trace()
# subj_list_postfix = 'C_single'

def create_folds(save_path, subj_data):
    subj_list_postfix = 'C'
    subj_list_dict = {'C':[]}
    for sub in subj_data.keys():
        subj_list_dict['C'].append(sub)
    
    subj_list = []
    

    for fold in range(5):
        for class_name in ['C']:
            class_list = subj_list_dict[class_name]
            np.random.shuffle(class_list)
            num_class = len(class_list)

            class_test = class_list[fold*int(0.2*num_class):(fold+1)*int(0.2*num_class)]
            class_train_val = class_list[:fold*int(0.2*num_class)] + class_list[(fold+1)*int(0.2*num_class):]
            class_val = class_train_val[:int(0.1*len(class_train_val))]
            class_train = class_train_val[int(0.1*len(class_train_val)):]
            subj_test_list = []
            subj_val_list = []
            subj_train_list = []
            subj_test_list.extend(class_test)
            subj_train_list.extend(class_train)
            subj_val_list.extend(class_val)

        if 'single' in subj_list_postfix:
            subj_id_list_train, case_id_list_train = get_subj_single_case_id_list(subj_data, subj_train_list)
            subj_id_list_val, case_id_list_val = get_subj_single_case_id_list(subj_data, subj_val_list)
            subj_id_list_test, case_id_list_test = get_subj_single_case_id_list(subj_data, subj_test_list)

            save_single_data_txt('../data/LAB/fold'+str(fold)+'_train_' + subj_list_postfix + '.txt', subj_id_list_train, case_id_list_train)
            save_single_data_txt('../data/LAB/fold'+str(fold)+'_val_' + subj_list_postfix + '.txt', subj_id_list_val, case_id_list_val)
            save_single_data_txt('../data/LAB/fold'+str(fold)+'_test_' + subj_list_postfix + '.txt', subj_id_list_test, case_id_list_test)
        else:
            subj_id_list_train, case_id_list_train = get_subj_pair_case_id_list(subj_data, subj_train_list)
            subj_id_list_val, case_id_list_val = get_subj_pair_case_id_list(subj_data, subj_val_list)
            subj_id_list_test, case_id_list_test = get_subj_pair_case_id_list(subj_data, subj_test_list)

            save_pair_data_txt(f'{save_path}fold{str(fold)}_train_{subj_list_postfix}.txt', subj_id_list_train, case_id_list_train)
            save_pair_data_txt(f'{save_path}fold{str(fold)}_val_{subj_list_postfix}.txt', subj_id_list_val, case_id_list_val)
            save_pair_data_txt(f'{save_path}fold{str(fold)}_test_{subj_list_postfix}.txt', subj_id_list_test, case_id_list_test)

if __name__ == "__main__":
    seed = 10
    np.random.seed(seed)

    # preprocess subject label and data --> (no labels for a healthy cohort)
    tsv_path = '/media/andjela/SeagatePor1/LSSL/data/participants.tsv'      
    data_path = '/media/andjela/SeagatePor1/LSSL/data/'
    # file_ending = 'T1w_64.nii.gz'
    file_ending = 'T1w_128.nii.gz'
    
    subj_data = create_subj_data(data_path, file_ending, tsv_path)

    h5_noimg_path = '/media/andjela/SeagatePor1/LSSL/data/CP_longitudinal_noimg.h5'
    save_subj_data_h5(h5_noimg_path, subj_data)
    
    h5_img_path = '/media/andjela/SeagatePor1/LSSL/data/CP_longitudinal_img.h5'
    # h5_img_path = '/media/andjela/SeagatePor1/LSSL/data/CP_longitudinal_img_crop.h5'
    # save_imgs_h5(h5_img_path, subj_data)

    h5_img_path_aug = '/media/andjela/SeagatePor1/LSSL/data/CP_longitudinal_img_aug.h5'
    aug_size = 10
    # generate_aug_data(h5_img_path_aug, aug_size, subj_data)

    save_path = '/media/andjela/SeagatePor1/LSSL/data/'
    # create_folds(save_path, subj_data)

    

    
