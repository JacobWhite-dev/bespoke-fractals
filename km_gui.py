import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
from matplotlib import pyplot as plt
from skimage import io
import time
import scipy.fftpack as fftpack

class KMGUI(tk.Tk):
    '''
    Kaleidoscope-multiplication theorem graphical input application class
    '''

    def __init__(self, imgPath = '9.gif', 
                 title = 'Kaleidoscope-Multiplication Theorem GUI'):

        super().__init__()

        # General
        self.title(title)

        if imgPath == None:
            data = np.zeros((512, 512), dtype = np.int)
            for r in range(0, 512):
                for c in range(0, 512):
                    if np.abs(r - 256) * np.abs(r - 256) + np.abs(c-256) * np.abs(c - 256) < 400:
                        data[r, c] = 1


            self._image = Image.fromarray(data)
        else:
            self._image = Image.open(imgPath)

        img = np.asarray(self._image)
        self._imgs = []
        for i in range(0, 512):
            self._imgs.append(io.imread("scaled_images/{}.png".format(i)))

        self._display_image = self._image

        self._render = ImageTk.PhotoImage(self._display_image)

        self._image_disp = tk.Label(self, image = self._render)
        self._image_disp.img = self._render
        self._image_disp.pack(side = 'top')

        #imgType = tk.IntVar()

        #self._radio = tk.Radiobutton(self, text='Circle', variable = imgType, value = 0,  indicatoron = 0)
        #self._radio.pack(side = 'top')
        #self._radio2 = tk.Radiobutton(self, text='Image', variable = imgType, value = 1,  indicatoron = 0)
        #self._radio2.pack(side = 'top')

        self._label = tk.Label(text = "Multiplication Factor")
        self._label.pack(side = 'top', expand = False, fill = tk.NONE)

        self._slider = tk.Scale(self, from_ = 0, to = 511, 
                                orient = tk.HORIZONTAL, command = self.update_image)
        self._slider.set(1)
        self._slider.pack(side = 'top', expand = True, fill = tk.X)

    def update_image(self, event):
        self._display_image = Image.fromarray(self._imgs[int(event)])
        self._render = ImageTk.PhotoImage(self._display_image)
        self._image_disp.config(image = self._render)
        self._image_disp.img = self._render

if __name__ == "__main__":
    app = KMGUI()
    app.mainloop()