import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class getbbox(object):

    def __init__(self, img,n):
        self.img = img
        self.n = n
        self.fig = plt.gcf()
        self.ax = plt.gca()
        self.x = [1]
        self.x_list = []
        self.y_list = []
        self.y = [2]
        self.fig.canvas.mpl_connect('button_press_event', self.button_press_event)
        plt.show()

    def button_press_event(self, event):
        if event.inaxes:
            x, y = event.xdata, event.ydata
            self.line = plt.Line2D([x, x], [y, y], marker='o')
            self.ax.add_line(self.line)
            self.fig.canvas.draw()
            self.x_list.append(x)
            self.y_list.append(y)
            if(len(self.x_list)==4*self.n):
                plt.close()

    def bbox_gen(self):
        bbox = np.zeros((self.n,4,2))
        for i in range(self.n):
            bbox[i,:,0] =self.x_list[4*i:(4*i+4)]
            bbox[i, :, 1] = self.y_list[4 * i:(4 * i + 4)]
        return bbox

def get_bbox(img,n):
    plt.imshow(img)
    draw = getbbox(img,n)
    bbox = draw.bbox_gen()
    return bbox.astype(int)

if __name__ =="__main__":
    image_file = "easyFrames/easy0.jpg"
    img = Image.open(image_file).convert('RGB')
    img = np.array(img)
    n=2
    bbox = get_bbox(img,n)
    print(bbox)
