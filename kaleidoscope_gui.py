import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
from matplotlib import pyplot as plt

####################################
# KALEIDOSCOPE TRANSFORM FUNCTIONS #
####################################

def kal_round(x, sigma):
    '''
    Rounds x up or down depending on whether sigma is positive or negative
    '''
    if sigma <= 0:
        return int(np.floor(x))
    else:
        return int(np.ceil(x))

def kaleidoscope(img, nu, sigma):
    '''
    Perform a nu,sigma-Kaleidoscope transform on img
    '''

    img = img // np.abs(sigma)

    imgNew = np.zeros_like(img, dtype = int);

    h, w = img.shape

    rows = np.arange(h)
    cols = np.arange(w)

    for r in rows:
        for c in cols:

            m1 = np.mod(kal_round(h / nu, sigma) * np.mod(r, nu) + sigma * (r // nu), h)
            m2 = np.mod(kal_round(w / nu, sigma) * np.mod(c, nu) + sigma * (c // nu), w)

            imgNew[m1, m2] += img[r, c]

    return imgNew

##########################
# KALEIDOSCOPE GUI CLASS #
##########################

class KaleidoscopeGUI(tk.Tk):
    '''
    Kaleidoscope transform graphical input application class
    '''

    def __init__(self, imgPath = '9.gif', title = 'Kaleidoscope Transform GUI'):

        super().__init__()

        # General
        self.title(title)

        # Image Data
        self._image = Image.open(imgPath)
        self._render = ImageTk.PhotoImage(self._image)
        self._k_image = self._image
        self._render_k = ImageTk.PhotoImage(self._k_image)

        # Image Frame
        self._display_frame = tk.Frame(self)
        self._display_frame.pack(side = 'top')

        # Original Image
        self._original_image_frame = tk.Frame(self._display_frame)
        self._original_image_frame.pack(side = 'left')

        self._original_label = tk.Label(self._original_image_frame, 
                                        text = 'Original Image')
        self._original_label.pack(side = 'top', expand = True)

        self._original_image = tk.Label(self._original_image_frame, 
                                        image = self._render)
        self._original_image.image = self._render
        self._original_image.pack(side = 'bottom')


        # Kaleidoscope Transform
        self._kaleidoscope_image_frame = tk.Frame(self._display_frame)
        self._kaleidoscope_image_frame.pack(side = 'right')

        self._kaleidoscope_label = tk.Label(self._kaleidoscope_image_frame, 
                                            text = 'Kaleidoscope Transform')
        self._kaleidoscope_label.pack(side = 'top', expand = True)

        self._kaleidoscope_image = tk.Label(self._kaleidoscope_image_frame, 
                                            image = self._render)
        self._kaleidoscope_image.image = self._render_k
        self._kaleidoscope_image.pack(side = 'bottom')

        # Entries
        self._entry_frame = tk.Frame(self)
        self._entry_frame.pack(side = 'bottom')

        # Downsampling Factor Entry
        self._nu_frame = tk.Frame(self._entry_frame)
        self._nu_frame.pack(side = 'left', expand = True)
        self._nu_label = tk.Label(self._nu_frame, text = "Downsampling Factor")
        self._nu_label.pack(side = 'left', expand = False, fill = tk.NONE)
        self._nu_entry = tk.Entry(self._nu_frame)
        self._nu_entry.insert(0, "1")
        self._nu_entry.pack(side = 'right', expand = False, fill = tk.NONE)

        # Smear Factor Entry
        self._sigma_frame = tk.Frame(self._entry_frame)
        self._sigma_frame.pack(side = 'left', expand = True)
        self._sigma_label = tk.Label(self._sigma_frame, text = "Smear Factor")
        self._sigma_label.pack(side = 'left', expand = False, fill = tk.NONE)
        self._sigma_entry = tk.Entry(self._sigma_frame)
        self._sigma_entry.insert(0, "1")
        self._sigma_entry.pack(side = 'right', expand = False, 
                               fill = tk.NONE)

        # Transform Button
        self._transform_button = tk.Button(self._entry_frame, 
                                           text = "Transform", 
                                       command = self.transform)
        self._transform_button.pack(side = 'left', expand = False, 
                                    fill = tk.NONE)

    def transform(self):
        '''
        Perform and display Kaleidoscope transform of image
        '''

        # Get image data
        data = np.asarray(self._image)

        # Perform Kaleidoscope transform
        new_data = kaleidoscope(data, int(self._nu_entry.get()), 
                                int(self._sigma_entry.get()))

        # Save image data
        self._k_image = Image.fromarray(new_data)
        self._render_k = ImageTk.PhotoImage(self._k_image)

        # Display Kaleidoscope transformed image
        self._kaleidoscope_image.configure(image = self._render_k)
        self._kaleidoscope_image.image = self._render_k

if __name__ == "__main__":
    app = KaleidoscopeGUI()
    app.mainloop()