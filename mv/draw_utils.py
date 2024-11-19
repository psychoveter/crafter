import numpy as np
import cv2
from matplotlib import pyplot as plt
from numpy._typing import ArrayLike

from mv.const import objects, index_first_object, object_keys


def plot_image_grid(images, grid_size, figsize=(10, 10), titles=None):
    """
    Plot a grid of images.

    :param images: List or array of images in numpy array format
    :param grid_size: Tuple (rows, columns) indicating size of the grid
    :param image_shape: Tuple (height, width) indicating the shape of each image
    :param figsize: Tuple (width, height) indicating the size of the figure
    :param titles: List of titles for each subplot; if None, no titles will be displayed
    """
    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    for i, ax in enumerate(axes.flat):
        if i < len(images):
            image = images[i]
            ax.imshow(image)
            if titles:
                ax.set_title(titles[i])
        else:
            ax.axis('off')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def draw_image_grid(images: list):

    def v_line(width=2):
        return np.ones((img.shape[0], width, img.shape[2]), np.uint8) * 127
    def h_line(width=2):
        return np.ones((width, img.shape[1], img.shape[2]), np.uint8) * 127

    img = images[0]
    for i in range(1, len(images)):
        img = np.concatenate((img, v_line(), images[i]), axis=1)
    return img

def channel_to_img(tensor, channel, side_size=32):
    b, c, w, h = tensor.shape
    arr = tensor[0, channel].detach().numpy()
    arr *= 255
    arr = arr.astype(np.uint8)

    img = np.zeros((side_size * w, side_size * h,3), np.uint8)
    for i in range(w):
        for j in range(h):
            img[i*side_size:(i+1)*side_size, j*side_size:(j+1)*side_size, 0] = arr[i,j]


    return img.transpose(1,0,2)

def render_tensor_onehot(t, env, side_size = 32) -> ArrayLike:
    return render_nparr_onehot(
        t.detach().numpy().round(),
        env,
        side_size
    )

def render_nparr_onehot(arr, env, side_size = 32, objects_keys=None) -> ArrayLike:
    c, w, h = arr.shape
    textures = env._textures

    canvas = np.zeros((w*side_size, h*side_size, 3), np.uint8)

    arr_materials = arr[:index_first_object, :, :].argmax(axis=0)
    arr_objects = arr[index_first_object:, : , :].argmax(axis=0) + index_first_object

    for x in range(w):
      for y in range(h):
          texture_names = [ object_keys[arr_materials[x,y]], object_keys[arr_objects[x,y]] ]
          # print(f"({x},{y}): {texture_names}")
          for t in texture_names:
            if t.startswith("none-"):
                continue
            img = textures.get(name=t, size=[side_size, side_size])
            # print(img.shape)
            if img.shape[-1] == 4:
                img = img[..., :3]
            wx = x * side_size
            wy = y * side_size
            canvas[wx: wx + side_size, wy: wy + side_size] = img

    return canvas.transpose(1, 0, 2)

def render_channels(sample, x, y, side_size=32):
    c, w, h = sample.shape
    channel = sample[:,x,y].detach().numpy()
    channel[:index_first_object] /= channel[:index_first_object].sum()
    channel[index_first_object:] /= channel[index_first_object:].sum()
    canvas = np.zeros(
        shape = (side_size, side_size * len(objects) + len(objects) - 1, 3),
        dtype = np.uint8
    )

    for i in range(c):
        canvas[:, i * side_size + i: (i + 1) * side_size + i, 0] = (channel[i] * 255).astype(np.uint8)
        canvas[:, (i + 1) * side_size + i : (i + 1) * side_size + i + 1, :] = 255

    return canvas


# video tools

def render_tensor_films(first, second, env, side_size = 32):
    np_first = first.detach().numpy()
    np_second = second.detach().numpy()

    l, c, w, h = np_first.shape

    frames_first = [render_nparr_onehot(np_first[i], env, side_size) for i in range(l)]
    frames_second = [render_nparr_onehot(np_second[i], env, side_size) for i in range(l)]

    stacked = np.concatenate((frames_first, frames_second), axis=2)
    show_video_with_slider(stacked)

def render_film_np_onehot(arr, env, side_size = 32) -> ArrayLike:
    assert(len(arr.shape) == 4)
    l, c, w, h = arr.shape
    frames = [render_nparr_onehot(arr[i], env, side_size) for i in range(l)]
    show_video_with_slider(frames, width=side_size*w, height=side_size*h)

def show_video_with_slider(frames, width:int = 512, height:int = 512):
    """
    Displays a series of images as a video with a frame slider and optional scaling.

    Parameters:
    - frames: List or array of images (each of shape (w, h, c)).
    - scale: Scaling factor for resizing the frames (e.g., 0.5 for half size, 2.0 for double size).
    """
    # Number of frames
    num_frames, h, w, c = frames.shape
    ratio = w / h
    width = int(height * ratio)
    print(f"Frames shape: {frames.shape}, ratio: {ratio}")

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
