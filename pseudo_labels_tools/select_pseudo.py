import numpy as np
import os
data_dir = './infer/'
files = os.listdir(data_dir)
uncertainties = {}
for file in files:
    if not file.endswith('npz'):
        continue
    # print(file)
    read_data = np.load(data_dir+file)['probabilities']
    read_data = read_data[1]
    uncertainty = -np.sum(read_data*np.log(read_data))/np.sum(read_data>0.5)
    uncertainties[file.split('.')[0]] = uncertainty

s = sorted(uncertainties.items(), key=lambda x: x[1], reverse=True)
bound = 0.06
res_list = [tu for tu in s if tu[1]< bound]
val_list = [tu[1] for tu in s]
average = np.mean(val_list)
min = np.min(val_list)
max = np.max(val_list)
pseudo_li = [data_dir+tu[0]+'.nii.gz' for tu in res_list]
print(pseudo_li)
print('value span: ['+str(min)+', '+str(max)+']')
print('average: '+str(average))