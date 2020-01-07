import glob
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def make_10_imgs(file_name, img_ind):
    current_video_imgs = sorted(glob.glob(os.path.join(file_name, '*.jpg')))
    slice_10_imgs = []
    for i in range(0, len(current_video_imgs), int(len(current_video_imgs) / 9)):
        slice_10_imgs.append(current_video_imgs[i])
    
    merged_imgs = np.zeros((220,2200,3))
    for ind in range(len(slice_10_imgs)):
        im = np.array(Image.open(slice_10_imgs[ind]).resize((220,220), Image.ANTIALIAS))
        for i in range(220):
            for j in range(220):
                merged_imgs[i, j+ind*220, :] = im[i, j, :]
    merged_imgs = np.array(merged_imgs, dtype=np.uint8)  
    plt.imshow(merged_imgs)
    plt.axis('off')
    plt.show()
    #plt.savefig('merge_%d.png'%img_ind)
    print(img_ind)

if __name__ == '__main__':
    folder_filenames = sorted(glob.glob(os.path.join('data', '*')))
    for i in range(len(folder_filenames)):
        make_10_imgs(folder_filenames[i], i)