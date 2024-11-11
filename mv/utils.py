import numpy as np
import crafter
import torch
import random
from numpy._typing import ArrayLike

from mv.const import objects


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


            if not cell[0] or cell[0] == 'None':
                material = 'none-material'
            else:
                material = cell[0]

            if not cell[1] or cell[1] == 'None':
                obj = 'none-object'
            else:
                obj = cell[1].texture

            # print(f'In cell ({i},{j}): {material} {obj}')
            material_index = objects[material]["index"]
            obj_index = objects[obj]["index"]

            result[i][j][material_index] = 1
            result[i][j][obj_index] = 1


    result = np.array(result, dtype=np.uint8)
    result = result.transpose(2,1,0)
    return result




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


