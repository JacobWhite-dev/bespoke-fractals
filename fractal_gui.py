import tkinter as tk
import bespoke
import finite2
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class FractalGUI(tk.Tk):
    '''
    Bespoke fractal graphical input application class.
    '''

    def __init__(self, dim, title = "Fractal GUI"):

        super().__init__()

        # General
        self.title(title)               # Window title
        self._dim = dim                 # Size of canvas (dim x dim)
        self._origin = dim // 2         # Pixel value at origin
        self._points = []               # List of selected points on canvas
        self._mirrored_points = []      # Mirrors of selected points
        self._poly = None               # Polygon object
        self._first_angle = None        # Angle of first point drawn
        self._last_angle = None         # Angle of last point drawn
        self._done = False              # Flag for if polygon is complete
        self._valid = True              # Flag for if current point is valid
        self._dashes = {True: None,     # Options for polygon dashes, based on
                        False: [1, 1]}  #   value of self._done flag
        self._colours = {True: "green", # Options for polygon colours, based on
                         False: "red"}  #   self._valid flag

        # Canvas 
        self._plot = tk.Canvas(self, bg = 'white', 
                               width = self._dim, 
                               height = self._dim)
        self._plot.create_line(0, self._origin, 
                               self._dim, 
                               self._origin)
        self._plot.create_line(self._origin, 0, self._origin, self._dim)
        self._plot.pack(side = 'top')
        self._plot.bind('<Button-1>', self.canvas_click)
        self._plot.bind('<Double-Button-1>', self.canvas_double_click)
        self._plot.bind('<Button-3>', self.canvas_right_click)
        self._plot.bind('<Motion>', self.canvas_motion)
        self.bind('<Escape>', self.canvas_unfollow)

        # Entries
        self._button_frame = tk.Frame(self)
        self._button_frame.pack(side = 'bottom', expand = True, fill = tk.X)

        self._entry_frame = tk.Frame(self)
        self._entry_frame.pack(side = 'bottom')

        # N Entry
        self._n_frame = tk.Frame(self._entry_frame)
        self._n_frame.pack(side = 'left', expand = True)
        self._n_label = tk.Label(self._n_frame, text = "N = ")
        self._n_label.pack(side = 'left', expand = False, fill = tk.NONE)
        self._n_entry = tk.Entry(self._n_frame)
        self._n_entry.insert(0, "256")
        self._n_entry.pack(side = 'right', expand = False, fill = tk.NONE)

        # K Entry
        self._k_frame = tk.Frame(self._entry_frame)
        self._k_frame.pack(side = 'left', expand = True)
        self._k_label = tk.Label(self._k_frame, text = "k = ")
        self._k_label.pack(side = 'left', expand = False, fill = tk.NONE)
        self._k_entry = tk.Entry(self._k_frame)
        self._k_entry.insert(0, "1")
        self._k_entry.pack(side = 'right', expand = False, fill = tk.NONE)

        # Sigma Entry
        self._s_frame = tk.Frame(self._entry_frame)
        self._s_frame.pack(side = 'left', expand = True)
        self._s_label = tk.Label(self._s_frame, text = "sigma = ")
        self._s_label.pack(side = 'left', expand = False, fill = tk.NONE)
        self._s_entry = tk.Entry(self._s_frame)
        self._s_entry.insert(0, "all")
        self._s_entry.pack(side = 'right', expand = False, fill = tk.NONE)

        # Propagate TickBox
        self._propagate = tk.IntVar()
        self._propagate_frame = tk.Frame(self._entry_frame)
        self._propagate_frame.pack(side = 'left', expand = True)
        self._propagate_tick = tk.Checkbutton(self._propagate_frame, 
                                              text = 'propagate', 
                                              variable = self._propagate)
        self._propagate_tick.pack(side = 'left', expand = False, fill = tk.NONE)

        # Solid TickBox
        self._solid = tk.IntVar()
        self._solid_frame = tk.Frame(self._entry_frame)
        self._solid_frame.pack(side = 'left', expand = True)
        self._solid_tick = tk.Checkbutton(self._solid_frame, 
                                              text = 'solid', 
                                              variable = self._solid)
        self._solid_tick.pack(side = 'left', expand = False, fill = tk.NONE)

        # Clear Button
        self._clear_button = tk.Button(self._button_frame, text = "Clear", 
                                       command = self.clear_canvas)
        self._clear_button.pack(side = 'left', expand = True, fill = tk.X)

        # Generate Button
        self._gen_button = tk.Button(self._button_frame, text = "Generate", 
                                     command = self.generate_fractal)
        self._gen_button.pack(side = 'right', expand = True, fill = tk.X)

    def to_x(self, pix):
        '''
        Convert horizontal pixel value to cartesian x coordinate
        '''
        return pix - self._origin

    def from_x(self, x):
        '''
        Convert cartesian x coordinate to horizontal pixel value
        '''
        return x + self._origin

    def to_y(self, pix):
        '''
        Convert vertical pixel value to cartesian y coordinate
        '''
        return self._origin - pix

    def from_y(self, y):
        '''
        Convert cartesian y coordinate to vertical pixel value
        '''
        return self._origin - y

    def calculate_angle(self, x_pix, y_pix):
        '''
        Given the pixel values of a point, determine it's angle from the
        positive x-axis, measured anticlockwise
        '''
        return math.atan2(self.to_y(y_pix), self.to_x(x_pix)) % (2 * math.pi)

    def clear_canvas(self):
        '''
        Clear GUI canvas, resetting relevant variables
        '''
        # Delete the current polygon if there is one
        if self._poly is not None:
            self._plot.delete(self._poly)
            self._poly = None

        # Clear points and their mirrors
        self._points = []
        self._mirrored_points = []

        # Clear stored angles
        self._last_angle = None
        self._first_angle = None

        # Reset flags
        self._done = False
        self._valid = True

    def mirror_point(self, x_pix, y_pix):
        '''
        Mirror a point across the x and y axes
        '''
        return self.from_x(-self.to_x(x_pix)), self.from_y(-self.to_y(y_pix))

    def is_angle_valid(self, angle):
        '''
        Determine if a proposed angle for adding a new point is valie
        '''
        if self._first_angle is None or self._last_angle is None:
            return True

        true_angle = (angle - self._first_angle) % (2 * math.pi)
        true_last_angle = (self._last_angle - self._first_angle) % (2 * math.pi)

        big_enough = true_angle > true_last_angle
        small_enough = true_angle < math.pi
        return big_enough and small_enough

    def add_point(self, x, y, angle = None):
        '''
        Add a point to the polygon
        '''
        self._points.extend([x, y])
        self._mirrored_points.extend(list(self.mirror_point(x, y)))

        if angle is None:
            angle = self.calculate_angle(x, y)

        if self._first_angle is None:
            self._first_angle = angle
        
        self._last_angle = angle

    def remove_point(self):
        '''
        Remove most recent point from the polygon
        '''
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
        '''
        Draw the polygon on the canvas
        '''
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
        sigma = self._s_entry.get()

        if sigma != 'all':
            sigma = [int(i) for i in sigma.split(',')]

        # Change list to be correct
        points = self._points + self._mirrored_points
        input = []
        for i in range(len(points) // 2):
            input.append([self.to_x(points[2 * i]), self.to_y(points[2 * i + 1])])
        
        input = np.array(input)
        angles = np.mod(np.arctan2(input[:, 1], input[:, 0]), 2 * math.pi)

        # Sort inputs by angle so that polygon thing works
        indices = np.argsort(angles)
        input = input[indices]

        if not self._solid.get():
            angles = finite2.sortedFarey(N, N, K, sortBy = lambda p,q: bespoke.poly(p,q,input))
            fractal = finite2.fractalise(N, N, angles, smearing = sigma, propagate = self._propagate.get())

        else:
            angles = []
            for i in range(-N, N):
                for j in range(-N, N):
                    if np.abs(bespoke.poly(j, i, input)) < 0.1:
                        angles.append(i + 1j * j)
            fractal = finite2.fractalise(N, N, angles, smearing = sigma, propagate = self._propagate.get())

        if self._propagate.get():
            fig = plt.figure()
            ims = []
            for img in fractal:
                im = plt.imshow(img, animated = True)
                ims.append([im])
               
            interval = 5000 // len(ims)
            ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=True,
                                repeat = False)
            plt.axis("off")
            plt.show()

        else:
            plt.imshow(fractal)
            plt.axis("off")
            plt.show()

if __name__ == "__main__":
    app = FractalGUI(600)
    app.mainloop()

