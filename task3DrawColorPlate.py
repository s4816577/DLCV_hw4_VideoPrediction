import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

COLOR_POOL = ('black', 'orange', 'purple', 'red', 'sienna', 'green', 'blue', 'grey', 'pink', 'yellow', 'cyan') 
TEXT_POOL = ('Other', 'Inspect/Read', 'Open', 'Take', 'Cut', 'Put', 'Close', 'Move Around', 'Divide/Pull Apart', 'Pour', 'Transfer')
    
def draw():    
    #parameters
    box_h = 0.05
    box_w = 0.05
    box_cur_x = 0.0
    box_cur_y = 0.5
    text_shift_x = 0.055
    text_shift_y = 0.01
    
    #make blank figure
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(111)
    
    #add Rectangles and Text
    for i in range(11):
        ax.add_patch(plt.Rectangle((box_cur_x, box_cur_y), box_w, box_h, color=COLOR_POOL[i]))
        ax.text(box_cur_x + text_shift_x, box_cur_y + text_shift_y, TEXT_POOL[i])
        box_cur_x += box_w + 0.12
        
        #adjust pose of second line
        if i == 5:
            box_cur_y -= 0.2
            box_cur_x = 0.0
    
    plt.axis('off')
    plt.show()
    #plt.savefig('video_predict_%d.png'%ind)

if __name__ == '__main__':
    draw()
