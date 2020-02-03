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

#print("Starting")

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

#def mirror_point(x, y):
#
#    return [-x, -y]

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
        self._first_angle = None
        self._last_angle = None
        self._done = False
        self._valid = True
        self._dashes = {True: None, False: [1, 1]}
        self._colours = {True: "green", False: "red"}

        # Canvas
        self._plot = tk.Canvas(self, bg = 'white', width = CANVAS_DIM, height = CANVAS_DIM)
        self._plot.create_line(0, CANVAS_ORIGIN, CANVAS_DIM, CANVAS_ORIGIN)
        self._plot.create_line(CANVAS_ORIGIN, 0, CANVAS_ORIGIN, CANVAS_DIM)
        self._plot.pack(side = 'top')
        self._plot.bind('<Button-1>', self.canvas_click)
        self._plot.bind('<Double-Button-1>', self.canvas_double_click)
        self._plot.bind('<Button-3>', self.canvas_right_click)
        self._plot.bind('<Motion>', self.canvas_motion)
        self.bind('<Escape>', self.canvas_unfollow)
        

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

        # Clear button
        self._clear_button = tk.Button(self._button_frame, text = "Clear", 
                                       command = self.clear_canvas)
        self._clear_button.pack(side = 'left', expand = False, fill = tk.NONE)

        # Generate Button
        self._gen_button = tk.Button(self._button_frame, text = "Generate", 
                                     command = self.generate_fractal)
        self._gen_button.pack(side = 'right', expand = False, fill = tk.NONE)

    def clear_canvas(self):
        if self._poly is not None:
            self._plot.delete(self._poly)
            self._poly = None
        self._points = []
        self._mirrored_points = []
        self._last_angle = None
        self._first_angle = None
        self._done = False
        self._valid = True

    def calculate_angle(self, x, y):
        return math.atan2(to_y(y), to_x(x)) % (2 * math.pi)

    def mirror_point(self, x, y):
        return from_x(-to_x(x)), from_y(-to_y(y))

    def is_angle_valid(self, angle):
        if self._first_angle is None or self._last_angle is None:
            return True

        true_angle = (angle - self._first_angle) % (2 * math.pi)
        true_last_angle = (self._last_angle - self._first_angle) % (2 * math.pi)

        big_enough = true_angle > true_last_angle
        small_enough = true_angle < math.pi
        return big_enough and small_enough

    def add_point(self, x, y, angle = None):
        self._points.extend([x, y])
        self._mirrored_points.extend(list(self.mirror_point(x, y)))

        if angle is None:
            angle = self.calculate_angle(x, y)

        if self._first_angle is None:
            self._first_angle = angle
        
        self._last_angle = angle

    def remove_point(self):

        if len(self._points) < 2:
            print("No points to remove")
            return

        del self._points[-2:]
        del self._mirrored_points[-2:]

        if len(self._points) < 2:
            self._first_angle = None
            self._last_angle = None
            return

        last_x = self._points[-2]
        last_y = self._points[-1]
        angle = self.calculate_angle(last_x, last_y)

        if len(self._points) < 4:
            self._first_angle = angle

        self._last_angle = angle

    def draw_poly(self, points, fill, outline, dash):
        if self._poly is not None:
            self._plot.delete(self._poly)

        if len(points) >= 4:
            self._poly = self._plot.create_polygon(points, fill = fill, outline = outline, dash = dash)

    def update_poly(self, points):
        self.draw_poly(points, "", self._colours.get(self._valid), self._dashes.get(self._done))

    def canvas_click(self, event):
        if self._done:
            self._done = False
            return

        angle = self.calculate_angle(event.x, event.y)
        self._valid = self.is_angle_valid(angle)

        if self._valid:
            self.add_point(event.x, event.y)
            self.update_poly(self._points + self._mirrored_points)

    def canvas_double_click(self, event):
        self.canvas_unfollow(event)    

    def canvas_right_click(self, event):
        if self._done:
            self._done = False
            return

        self.remove_point()
        self.update_poly(self._points + self._mirrored_points)

    def canvas_unfollow(self, event):
        self._done = True
        self._valid = True
        self.update_poly(self._points + self._mirrored_points)

    def canvas_motion(self, event):

        if self._done:
            return

        angle = self.calculate_angle(event.x, event.y)
        self._valid = self.is_angle_valid(angle)

        points = self._points + [event.x, event.y] + self._mirrored_points + list(self.mirror_point(event.x, event.y))
        self.update_poly(points)

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
        angles = np.mod(np.arctan2(input[:, 1], input[:, 0]), 2 * math.pi)

        # Sort inputs by angle so that polygon thing works
        indices = np.argsort(angles)
        print(angles)
        print(input)
        input = input[indices]
        print(input)

        lines, angles, mValues, fractal, overSamplingFilter = bespoke.myFiniteFractal(N, K, sortBy = lambda p,q: bespoke.poly(p,q,input), twoQuads = True)
        fractal = fftpack.fftshift(fractal)
        plt.imshow(fractal)
        plt.show()


if __name__ == "__main__":
    app = FractalGUI()
    app.mainloop()

