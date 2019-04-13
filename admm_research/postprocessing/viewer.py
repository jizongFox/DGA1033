# this is the viewer script for 3D volumns visualization
import matplotlib.pyplot as plt
from skimage import data

astronaut = data.astronaut()
ihc = data.immunohistochemistry()
hubble = data.hubble_deep_field()

# Initialize the subplot panels side by side



import tempfile
import os
# Create a temporary directory
try:
    d = os.mkdir('file',)
except FileExistsError:
    d= 'file'
import os

# Return the tail of the path
os.path.basename('http://google.com/attention.zip')
from urllib.request import urlretrieve

# Define URL
url = 'http://www.fil.ion.ucl.ac.uk/spm/download/data/attention/attention.zip'

# Retrieve the data
# fn, info = urlretrieve(url, os.path.join(d, 'attention.zip'))
# import zipfile

# Extract the contents into the temporary directory we created earlier
# zipfile.ZipFile(fn).extractall(path=d)

from skimage import io
import nibabel
# struct_arr = io.imread("https://s3.amazonaws.com/assets.datacamp.com/blog_assets/attention-mri.tif")
struct_arr = nibabel.load('file/attention/functional/snffM00587_0023.hdr').get_data()
struct_arr = struct_arr.T

struct_arr2 = nibabel.load('file/attention/functional/snffM00587_0030.hdr').get_data()
struct_arr2 = struct_arr2.T


def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)
def multi_slice_viewer(volumes):
    # remove_keymap_conflicts({'j', 'k'})
    fig, axs = plt.subplots(1,2,)

    for i, (ax,volume) in enumerate(zip(axs,volumes)):
        ax.volume = volume
        ax.index = volume.shape[0] // 2
        ax.imshow(volume[ax.index])
        ax.set_title(f'plane = {ax.index}')
        ax.axis('off')

    fig.canvas.mpl_connect('key_press_event', process_key)
    fig.canvas.mpl_connect('scroll_event', process_mouse_wheel)

def process_mouse_wheel(event):
    fig = event.canvas.figure
    for i, ax in enumerate(fig.axes):
        if event.button == 'up':
            previous_slice(ax)
        elif event.button == 'down':
            next_slice(ax)
    fig.canvas.draw()

def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()

def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) if (ax.index-1) >=0 else 0  # wrap around using %
    ax.images[0].set_array(volume[ax.index])
    ax.set_title(f'plane = {ax.index}')

def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) if (ax.index + 1) <  volume.shape[0] else volume.shape[0]-1
    ax.images[0].set_array(volume[ax.index])
    ax.set_title(f'plane = {ax.index}')


multi_slice_viewer([struct_arr,struct_arr2])
plt.show()