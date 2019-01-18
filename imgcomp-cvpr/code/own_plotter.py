
import os
import matplotlib.pyplot as plt
import csv
from own_utils import get_job_ids
import numpy as np
#OUR CODE

weight = 1000
suffix = ''
#THIS has to be modified
folder = '/srv/glusterfs/cmatija/fab-jul/data/logdir_b_5_weight_'+str(weight)+suffix
dset_name = 'kodak'
csv_name = 'measures.csv'
plot_file_type = '.pdf'
job_ids = get_job_ids(folder, mode='plot')
mssim_ind = 2
bpp_ind = 1
psnr_ind = 3
data={}
labels = []
for job in job_ids:
    job_ids = job[0].split(',')
    label = job[1]
    labels.append(label)
    mssim_means = []
    psnr_means = []
    bpp_means = []
    for config in job_ids:
        folder_name = config + ' ' + dset_name
        mssims = []
        bpps = []
        psnrs = []
        with open(os.path.join(folder,folder_name, csv_name), 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            for row_ind, row in enumerate(reader):
                if row_ind == 0:
                    continue
                mssims.append(float(row[mssim_ind]))
                bpps.append(float(row[bpp_ind]))
                psnrs.append(float(row[psnr_ind]))
        mssim_means.append(np.mean(mssims))
        bpp_means.append(np.mean(bpps))
        psnr_means.append(np.mean(psnrs))
    data[label] = (bpp_means, mssim_means, psnr_means)
lines = []

#plot msssim
f = plt.figure()
for label in labels:
    plt.plot(data[label][0], data[label][1], marker='x', label=label)
plt.ylim(0.8, 1.02)
plt.xlim(0.15, 1.2)
plt.ylabel('MS-SSIM', fontsize=15)
plt.xlabel('bpp', fontsize=15)
plt.legend(loc='lower right', fontsize=15)
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=13)
ax.tick_params(axis='both', which='minor', labelsize=13)
plt.grid()
plt.show()
f.savefig(str(weight)+'_msssim_' +plot_file_type, bbox_inches='tight')
plt.close()
#plot psnr
f = plt.figure()
for label in labels:
    plt.plot(data[label][0], data[label][2], marker='x', label=label)
plt.ylim(15, 35)
plt.xlim(0.15, 1.2)
plt.ylabel('psnr', fontsize=15)
plt.xlabel('bpp', fontsize=15)
# plt.legend(loc='lower right', fontsize=15)
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=13)
ax.tick_params(axis='both', which='minor', labelsize=13)
plt.grid()
plt.show()
f.savefig(str(weight)+'_psnr_' +plot_file_type, bbox_inches='tight')