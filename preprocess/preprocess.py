import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def preprocessing(raw, i, redo=False):
    if os.path.exists(temp_dir + str(i) + '_icaed.fif') and redo==False:
        raw_icaed = mne.io.read_raw_fif (temp_dir + str(i) + '_icaed.fif') 
        return raw_icaed
    else:
        montage = mne.channels.make_standard_montage('standard_1020')
        # montage.plot()
        # plt.show()
        raw.set_montage(montage=montage, match_case=False)
        raw = raw.set_eeg_reference(ref_channels='average')
        while True:
            raw.plot()
            ica = mne.preprocessing.ICA(n_components=10, random_state=97, max_iter=800)
            ica.fit(raw)
            ica.plot_components()
            plt.show()

            input_channels = input('输入1个坏道名称，否则输入none：')
            print(input_channels)
            if input_channels=='none':
                break
            else:
                raw.info['bads'].append(input_channels)
                raw = raw.interpolate_bads(reset_bads=False)
                print(raw.info['bads'])

        ica.fit(raw)
        ica.plot_components()
        ica.exclude = [int(x) for x in input("请输入不要的ICA成分，空格分开：").split()]
        ica.apply(raw)
        raw.save(temp_dir + str(i) + '_icaed.fif', overwrite=True)

        return raw


# 单位uV
multi_personal_data = np.load('data_all_4_47_ds.npy').transpose(0, 2, 1)*10**(-6)

# mne.info创建
info = mne.create_info(
    ch_names=['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 'FC6', 'Cz', 'C3', 'C4', 'T7', 'T8', 'CP1', 'CP2', 'CP5', 'CP6', 'Pz', 'P3', 'P4', 'P7', 'P8', 'PO3', 'PO4', 'Oz', 'O1', 'O2', 'A2', 'A1'], 
    ch_types=['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg','eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg','eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg'], 
    sfreq=125
)

# 视频时长
vedio_length = [81, 63, 73, 78, 69, 90, 56, 60, 105, 45, 60, 82, 35, 44, 38, 43, 55, 69, 73, 129, 77, 75, 34, 37, 67, 63, 54, 77]

# 80个被试，28个视频片段，使用30s数据时长，32个eeg通道（可以改为33以增加时间戳用于检查）
used_data = np.zeros((80, 28, 30*125, 32))

# 创建trial时间戳
data_index = np.zeros_like(vedio_length)
for i in range(28):
    data_index[i] = np.sum(vedio_length[:i+1])*125-1

# 临时fif文件储存地址
temp_dir = 'data/subject_'

# 创建待使用数据，可以调整开始人工鉴别的数据起始点
for i in range(80):
    subject_raw = mne.io.RawArray(multi_personal_data[i, :, :], info)
    # 预处理算法
    raw_icaed = preprocessing(subject_raw, i, redo=True)
    df = raw_icaed.to_data_frame()
    for j in range(28):
        # 要检查时间列，把下一行末尾由1：改为0：
        used_data[i, j, :, :] = df.values[data_index[j]-30*125:data_index[j], 1:]

X_total = used_data
y_total = np.array((
    [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8],
))

np.save('X_total', X_total)
np.save('y_toral', y_total.repeat(80, axis=0))


