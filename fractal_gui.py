import tkinter as tk
import bespoke
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack
import pyfftw

# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

# Turn on the cache for optimum performance
pyfftw.interfaces.cache.enable()

print("Starting")

CANVAS_DIM = 500
CANVAS_ORIGIN = CANVAS_DIM // 2

def to_x(x):
    return x - CANVAS_ORIGIN

def from_x(x):
    return x + CANVAS_ORIGIN

def to_y(y):
    return CANVAS_ORIGIN - y

def from_y(y):
    return CANVAS_ORIGIN - y

def mirror_point(x, y):

    return [-x, -y]

def on_click(event):
    #print(event.x, event.y)
    x = to_x(event.x)
    y = to_y(event.y)
    points.append([x, y])
    points.append(mirror_point(x,y))
    canvas.create_polygon(map(lambda point: [from_x(point[0]), from_y(point[1])], points))

class FractalGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Fractal GUI")
        self._points = []
        self._mirrored_points = []
        self._poly = None
        self._arc1 = None
        self._arc2 = None
        self._last_angle = 0
        self._done = False

        # Canvas
        self._plot = tk.Canvas(self, bg = 'white', width = CANVAS_DIM, height = CANVAS_DIM)
        self._plot.create_line(0, CANVAS_ORIGIN, CANVAS_DIM, CANVAS_ORIGIN)
        self._plot.create_line(CANVAS_ORIGIN, 0, CANVAS_ORIGIN, CANVAS_DIM)
        self._plot.pack(side = 'top')
        self._plot.bind('<Button-1>', self.canvas_click)
        self._plot.bind('<Double-Button-1>', self.canvas_double_click)
        self._plot.bind('<Button-3>', self.canvas_right_click)
        self._plot.bind('<Motion>', self.canvas_motion)

        # Entries
        self._button_frame = tk.Frame(self)
        self._button_frame.pack(side = 'bottom')

        # N Entry
        self._n_frame = tk.Frame(self._button_frame)
        self._n_frame.pack(side = 'left', expand = True)
        self._n_label = tk.Label(self._n_frame, text = "N = ")
        self._n_label.pack(side = 'left', expand = False, fill = tk.NONE)
        self._n_entry = tk.Entry(self._n_frame)
        self._n_entry.insert(0, "256")
        self._n_entry.pack(side = 'right', expand = False, fill = tk.NONE)

        # K Entry
        self._k_frame = tk.Frame(self._button_frame)
        self._k_frame.pack(side = 'left', expand = True)
        self._k_label = tk.Label(self._k_frame, text = "k = ")
        self._k_label.pack(side = 'left', expand = False, fill = tk.NONE)
        self._k_entry = tk.Entry(self._k_frame)
        self._k_entry.insert(0, "1")
        self._k_entry.pack(side = 'right', expand = False, fill = tk.NONE)

        # Generate Button
        self._gen_button = tk.Button(self._button_frame, text = "Generate", 
                                     command = self.generate_fractal)
        self._gen_button.pack(side = 'right', expand = False, fill = tk.NONE)

    def canvas_click(self, event):
        self._done = False

        x = to_x(event.x)
        y = to_y(event.y)
        #self._plot.create_line(event.x, event.y, from_x(-x), from_y(-y))
        if self._poly is not None:
            self._plot.delete(self._poly)

        self._points.extend([event.x, event.y])
        self._mirrored_points.extend([from_x(-x), from_y(-y)])
        try:
            self._poly = self._plot.create_polygon(self._points + self._mirrored_points, fill = "", outline = "black")
        except:
            pass

        self._last_angle = math.atan2(y,x) * 180 / math.pi

    def canvas_double_click(self, event):
        self.canvas_click(event)
        self._done = True

    def canvas_right_click(self, event):
        self._done = False

        try:
            self._points = self._points[:-2]
            self._mirrored_points = self._mirrored_points[:-2]
        except IndexError:
            pass

        if self._poly is not None:
            self._plot.delete(self._poly)

        try:
            self._poly = self._plot.create_polygon(self._points + self._mirrored_points, fill = "", outline = "black")
        except:
            pass

    def canvas_motion(self, event):

        if self._done:
            return

        fills = {True: 'green', False: 'red'}
        x = to_x(event.x)
        y = to_y(event.y)
        #r = math.sqrt(x * x + y * y)
        r = 25
        theta = math.atan2(y,x) * 180 / math.pi

        fill = fills[theta > self._last_angle]

        if self._arc1 is not None:
            self._plot.delete(self._arc1)
        if self._arc2 is not None:
            self._plot.delete(self._arc2)

        #self._arc1 = self._plot.create_arc(250 + r, 250 + r, 250 - r, 250 - r, start = self._last_angle, extent = theta - self._last_angle, fill = fill)
        #self._arc2 = self._plot.create_arc(250 + r, 250 + r, 250 - r, 250 - r, start = 180 + self._last_angle, extent = theta - self._last_angle, fill = fill)
        #self._arc1 = self._plot.create_arc(250 + r, 250 + r, 250 - r, 250 - r, start = 0, extent = theta, fill = fill)
        #self._arc1 = self._plot.create_line(from_x(-x), from_y(-y), event.x, event.y, dash = [1,1], fill = fill)

        self._arc2 = self._plot.create_polygon(self._points + [event.x, event.y] + self._mirrored_points + [from_x(-x), from_y(-y)], dash = [1,1], fill = "",
                                               outline = fill)
        #self._last_angle = theta

    def generate_fractal(self):
        # Get N and K
        N = int(self._n_entry.get())
        K = int(self._k_entry.get())

        # Change list to be correct
        points = self._points + self._mirrored_points
        print(points)
        input = []
        for i in range(len(points) // 2):
            input.append([to_x(points[2 * i]), to_y(points[2 * i + 1])])
        
        print(input)
        input = np.array(input)

        lines, angles, mValues, fractal, overSamplingFilter = bespoke.myFiniteFractal(N, K, sortBy = lambda p,q: bespoke.poly(p,q,input), twoQuads = True)
        fractal = fftpack.fftshift(fractal)
        plt.imshow(fractal)
        plt.show()


if __name__ == "__main__":
    app = FractalGUI()
    app.mainloop()

