#%%
import os
import numpy as np
import random

import crafter

ds_human_path = '/Users/Oleg.Bukhvalov/projects/montevideo/crafter/crafter_human_dataset'
ds_human_datasets_path = os.path.join(ds_human_path, 'dataset')
ds_human_file_paths = [f"{ds_human_datasets_path}/{i}" for i in os.listdir(ds_human_datasets_path)]

n0 = np.load(ds_human_file_paths[0])


import torch
from torch.utils.data import DataLoader, Dataset

class CrafterHumanDataset(Dataset):

    def __init__(self,
                 path: str,
                 film_length: int = 32,
                 dataset_size: int = 100
                 ):
        super().__init__()
        self.ds_human_datasets_path = os.path.join(ds_human_path, 'dataset')
        self.ds_human_file_paths = [f"{ds_human_datasets_path}/{i}" for i in os.listdir(ds_human_datasets_path)]
        self.dataset_size = dataset_size
        self.film_length = film_length
        self.films_per_file = self.dataset_size / len(self.ds_human_datasets_path)

        self.films = []

        temp_files = self.ds_human_file_paths.copy()

        while len(self.films) < self.dataset_size:
            file_index = random.randint(0, len(temp_files) - 1)
            file_path = temp_files.pop(file_index)
            with open(file_path, 'rb') as file:
                print(f"open file {file_path}")
                npz_file = np.load(file)
                films_from_file = 0
                while films_from_file < self.films_per_file:
                    item = self._parse_item(npz_file)
                    self.films.append(item)
                    films_from_file += 1
            print(f"Loaded {len(self.films)}")

    def __len__(self):
        return len(self.films)

    def __getitem__(self, idx):
        return self.films[idx]



class CrafterHumanDataset3DImages(CrafterHumanDataset):
    def __init__(self,
                 path: str,
                 film_length: int = 32,
                 dataset_size: int = 100):
        super().__init__(path, film_length, dataset_size)

    def _parse_item(self, npz_file):
        images = npz_file['image']
        i_from = random.randint(0, len(images) - self.film_length - 1)
        return images[i_from:i_from + self.film_length]


class CrafterHumanDataset3DSemantic(CrafterHumanDataset):
    def __init__(self,path, film_length: int = 32, dataset_size: int = 100):
        super().__init__(path, film_length, dataset_size)

    def _parse_item(self, npz_file):
        smap = npz_file['semantic']
        i_from = random.randint(0, len(smap) - self.film_length - 1)
        return smap[i_from:i_from + self.film_length]


dataset = CrafterHumanDataset3DSemantic(ds_human_path)


#%%
import cv2

def show_video_with_slider(frames, width:int = 512, height:int = 512):
    """
    Displays a series of images as a video with a frame slider and optional scaling.

    Parameters:
    - frames: List or array of images (each of shape (w, h, c)).
    - scale: Scaling factor for resizing the frames (e.g., 0.5 for half size, 2.0 for double size).
    """
    # Number of frames
    num_frames = len(frames)

    # Window to display frames
    cv2.namedWindow("Video with Slider")

    # Trackbar callback function to display the selected frame
    def on_trackbar(val):
        frame = frames[val]  # Get the frame corresponding to the slider position

        # Resize the frame based on the scale
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

        frame = frame.astype(np.uint8)  # Ensure it's in uint8 format for display
        cv2.imshow("Video with Slider", frame)

    # Create a trackbar in the window
    cv2.createTrackbar("Frame", "Video with Slider", 0, num_frames - 1, on_trackbar)

    # Initial display of the first frame
    on_trackbar(0)

    # Wait until the user presses 'q' to exit
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close the window
    cv2.destroyAllWindows()


frames = dataset.__getitem__(0)

show_video_with_slider(frames)