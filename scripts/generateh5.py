from PIL import Image
import os
import numpy as np
import h5py

def generateh5(scale_factor):
    scale = scale_factor
    num = "%06d"
    dir = ('datasets/STSR/train')
    h5_file = h5py.File(dir+'/Flickr1024_patches_x'+str(scale)+'.h5', 'w')
    idx_patch = 0
    for i in range(len(os.listdir(dir + "/HR"))//2):
        print('generating i%d'%(i+1))
        img_hr_0 = Image.open(dir + "/HR/"+str("%04d"%(i+1))+"_L.png")
        img_hr_1 = Image.open(dir + "/HR/"+str("%04d"%(i+1)) + "_R.png")
        img_lr_0 = Image.open(dir + "/LR/X"+str(scale)+str("/%04d"%(i+1))+"_L.png")
        img_lr_1 = Image.open(dir + "/LR/X"+str(scale)+str("/%04d"%(i+1)) + "_R.png")

        img_hr_0 = np.array(img_hr_0).astype(int)
        img_hr_1 = np.array(img_hr_1).astype(int)
        img_lr_0 = np.array(img_lr_0).astype(int)
        img_lr_1 = np.array(img_lr_1).astype(int)

        for x_lr in range(2, img_lr_0.shape[0]-34,20):
            for y_lr in range(2, img_lr_0.shape[1]-94,20):
                x_hr = x_lr*scale
                y_hr = y_lr*scale

                hr_patch_0=(img_hr_0[x_hr:(x_lr+30)*scale,y_hr:(y_lr+90)*scale,:]) 
                hr_patch_1=(img_hr_1[x_hr:(x_lr+30)*scale,y_hr:(y_lr+90)*scale,:])
                lr_patch_0=(img_lr_0[x_lr:x_lr+30,y_lr:y_lr+90,:])
                lr_patch_1=(img_lr_1[x_lr:x_lr+30,y_lr:y_lr+90,:])

                h5_file.create_dataset('hr_L/'+str(num % idx_patch), data=np.uint8(hr_patch_0),compression='gzip')
                h5_file.create_dataset('hr_R/'+str(num % idx_patch), data=np.uint8(hr_patch_1),compression='gzip')
                h5_file.create_dataset('lr_L/'+str(num % idx_patch), data=np.uint8(lr_patch_0),compression='gzip')
                h5_file.create_dataset('lr_R/'+str(num % idx_patch), data=np.uint8(lr_patch_1),compression='gzip')    
                idx_patch = idx_patch + 1

    h5_file.close()
    print(idx_patch)


if __name__ == '__main__':
    generateh5(2)
    generateh5(4)





