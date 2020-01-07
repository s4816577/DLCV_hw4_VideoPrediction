import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

COLOR_POOL = ('black', 'orange', 'purple', 'red', 'sienna', 'green', 'blue', 'grey', 'pink', 'yellow', 'cyan')

def parse_label(filename):
    result = []
    with open(filename, 'r', encoding='utf8') as f:
        for line in f:
            token = line.strip().split()
            if len(token) != 1:
                continue
            label = int(token[0])
            result.append(label)
    return result    
    
def draw(gt_labels, pre_labels, ind):
    #read img
    im = np.array(Image.open('merge_%d.png'%ind))
    
    #upper parameters
    upper_box_y = 255
    upper_box_h = 30
    upper_box_w = 1057/len(gt_labels)
    upper_box_cur_x = 171.0
    
    #downer parameters
    downer_box_y = 390
    downer_box_h = 30
    downer_box_w = 1057/len(pre_labels)
    downer_box_cur_x = 171.0
    
    #make blank figure and add img
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    #ax.set_xlim([0, 224])
    #ax.set_ylim([27, -6])
    ax.imshow(im)
    
    #add upper Rectangles
    for i in range(len(gt_labels)):
        ax.add_patch(plt.Rectangle((upper_box_cur_x, upper_box_y), upper_box_w, upper_box_h, color=COLOR_POOL[gt_labels[i]]))
        upper_box_cur_x += upper_box_w
        
    #add downer Rectangles
    for i in range(len(pre_labels)):
        ax.add_patch(plt.Rectangle((downer_box_cur_x, downer_box_y), downer_box_w, downer_box_h, color=COLOR_POOL[pre_labels[i]]))
        downer_box_cur_x += downer_box_w
    
    plt.axis('off')
    #plt.show()
    plt.savefig('video_predict_%d.png'%ind)
    print(ind)

if __name__ == '__main__':
    for i in range(7):
        gt_labels = parse_label('task3_gt_%d.csv'%i)
        pre_labels = parse_label('task3_pre_%d.csv'%i)
        draw(gt_labels, pre_labels, i)
