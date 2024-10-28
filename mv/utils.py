import numpy as np
from collections import OrderedDict
from matplotlib import pyplot as plt
import crafter
import torch
import random
from numpy._typing import ArrayLike


objects = OrderedDict([
    ("None",             { "index": 0,  "weight": 1 }),
    ("water",            { "index": 1,  "weight": 1 }),
    ("grass",            { "index": 2,  "weight": 1 }),
    ("stone",            { "index": 3,  "weight": 1 }),
    ("path",             { "index": 4,  "weight": 1 }),
    ("sand",             { "index": 5,  "weight": 1 }),
    ("tree",             { "index": 6,  "weight": 1 }),
    ("lava",             { "index": 7,  "weight": 1 }),
    ("coal",             { "index": 8,  "weight": 1 }),
    ("iron",             { "index": 9,  "weight": 1 }),
    ("diamond",          { "index": 10, "weight": 1 }),
    ("table",            { "index": 11, "weight": 1 }),
    ("furnace",          { "index": 12, "weight": 1 }),
    ('player-sleep',     { "index": 13, "weight": 1 }),
    ('player-left',      { "index": 14, "weight": 1 }),
    ('player-right',     { "index": 15, "weight": 1 }),
    ('player-up',        { "index": 16, "weight": 1 }),
    ('player-down',      { "index": 17, "weight": 1 }),
    ("cow",              { "index": 18, "weight": 1 }),
    ("zombie",           { "index": 19, "weight": 1 }),
    ("skeleton",         { "index": 20, "weight": 1 }),
    ('arrow-left',       { "index": 21, "weight": 1 }),
    ('arrow-right',      { "index": 22, "weight": 1 }),
    ('arrow-up',         { "index": 23, "weight": 1 }),
    ('arrow-down',       { "index": 24, "weight": 1 }),
    ('plant-ripe',       { "index": 25, "weight": 1 }),
    ('plant',            { "index": 26, "weight": 1 }),
    ("fence",            { "index": 27, "weight": 1 })
])

object_weights = [objects[x]['weight'] for x in objects]

def get_object_dict(obj):
    if obj is None:
        return objects["None"]
    else:
        return objects[obj]

def create_tensor_onehot(env, pos = None, view = np.array([9, 9])) -> torch.Tensor:
    np_sample = create_nparr_onehot(env, pos, view)
    torch_sample = torch.tensor(np_sample, dtype=torch.float32).unsqueeze(dim=0)
    return torch_sample

def create_nparr_onehot(env, pos = None, view = np.array([9, 9])) -> ArrayLike:
    """
    Creates onehot encoded 3d numpy array of shape C,H,W for current player position
    W,H - width, height of the view
    C - one hot channels for objects and materials from mv.utils.objects dictionary

    :param view: view size to generate
    :param env: Crafter Environment
    :return: 3d array CxHxW to match torch.nn.Conv2d input
    """

    world = env._world
    if pos is None:
        pos = env._player.pos

    idx = [i - view[0]//2 + pos[0] for i in range(view[0])]
    jdx = [j - view[1]//2 + pos[1] for j in range(view[1])]

    result = [None] * len(idx)
    for i in range(len(idx)):
        result[i] = [None] * len(jdx)
        for j in range(len(jdx)):
            result[i][j] = [0] * len(objects)

            cell = world[idx[i],jdx[j]]
            if not cell:
                continue

            material = cell[0]
            obj = cell[1].texture if cell[1] else None
            # print(f'In cell ({i},{j}): {material} {obj}')
            material_index = get_object_dict(material)["index"]
            obj_index = get_object_dict(obj)["index"]

            result[i][j][material_index] = 1
            if obj_index:
                result[i][j][obj_index] = 1

    result = np.array(result, dtype=np.uint8)
    result = result.transpose(2,1,0)
    return result



def render_tensor_onehot(t, env, side_size = 32) -> ArrayLike:
    return render_nparr_onehot(
        t.detach().numpy().round(),
        env,
        side_size
    )

def render_nparr_onehot(arr, env, side_size = 32) -> ArrayLike:
    c, w, h = arr.shape
    textures = env._textures
    items = list(objects.items())
    canvas = np.zeros((w*side_size, h*side_size, 3), np.uint8)
    for x in range(w):
      for y in range(h):
          texture_names = [ items[i][0] for i in range(c) if arr[i,x,y] > 0 ]
          # print(f"({x},{y}): {texture_names}")
          for t in texture_names:
            img = textures.get(name=t, size=[side_size, side_size])
            # print(img.shape)
            if img.shape[-1] == 4:
                img = img[..., :3]
            wx = x*side_size
            wy = y*side_size
            canvas[wx: wx + side_size, wy: wy + side_size] = img

    return canvas.transpose(1, 0, 2)


def draw_image_grid(images: list):
    img = images[0]
    for i in range(1, len(images)):
        line = np.ones((img.shape[0], 1, img.shape[2]), np.uint8) * 127
        img = np.concatenate((img, line, images[i]), axis=1)
    return img


def sample_nparr_onehot(num: int, env: crafter.Env, samples_from_world: int = 1000, view_size=9):
    env.reset()
    offset = view_size // 2
    def gen():
        current_samples = 0
        for i in range(num):
            current_samples += 1
            pos = [
                random.randint(offset, env._size[0] - offset),
                random.randint(offset, env._size[1] - offset)
            ]
            # env.step(env.action_space.sample())
            result = create_nparr_onehot(env, pos=pos, view=np.array([view_size, view_size]))
            if current_samples >= samples_from_world:
                current_samples = 0
                env.reset()
            yield result
    return list(gen())


def miss_rate(a1, a2):
    """
    Computes miss rate between two onehot grid arrays
    :param a1: C H W onehot array
    :param a2: C H W onehot array
    :return: sum of absolute differences / total number of elements
    """
    diff = a1.astype(np.int16) - a2.astype(np.int16)
    miss = np.abs(diff).sum()
    total = a1.shape[0] * a1.shape[1] * a1.shape[2]
    # print(f"miss: {miss} / total: {total} / miss rate: {miss / total}")
    return miss / total


def miss_rate_all(s1, s2):
    """
    Computes miss rate between two sequences of onehot grid arrays using miss_rate function
    :param s1:
    :param s2:
    :return:
    """
    assert len(s1) == len(s2)
    s = 0
    for i in range(len(s1)):
        s += miss_rate(s1[i], s2[i])
    return s


def get_actual_device():
    if torch.cuda.is_available():
        return torch.device('cuda')

    if torch.backends.mps.is_available():
        return torch.device('mps')

    return torch.device('cpu')



def plot_image_grid(images, grid_size, image_shape, figsize=(10, 10), titles=None):
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
            image = images[i] #.reshape(image_shape)
            ax.imshow(image, cmap='gray')
            if titles:
                ax.set_title(titles[i])
        else:
            ax.axis('off')
        ax.axis('off')

    plt.tight_layout()
    plt.show()