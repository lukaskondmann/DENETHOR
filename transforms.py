import numpy as np

def random_crop(image_stack, mask, image_size):
    H, W = image_stack.shape[2:]

    # skip random crop is image smaller than crop size
    if H - image_size // 2 <= image_size:
        return image_stack, mask
    if W - image_size // 2 <= image_size:
        return image_stack, mask

    h = np.random.randint(image_size, H - image_size // 2)
    w = np.random.randint(image_size, W - image_size // 2)

    image_stack = image_stack[:, :, h - int(np.floor(image_size // 2)):int(np.ceil(h + image_size // 2)),
                  w - int(np.floor(image_size // 2)):int(np.ceil(w + image_size // 2))]
    mask = mask[h - int(np.floor(image_size // 2)):int(np.ceil(h + image_size // 2)),
           w - int(np.floor(image_size // 2)):int(np.ceil(w + image_size // 2))]

    return image_stack, mask

def crop_or_pad_to_size(image_stack,  mask, image_size):
    T, D, H, W = image_stack.shape
    hpad = image_size - H
    wpad = image_size - W

    # local flooring and ceiling helper functions to save some space
    def f(x):
        return int(np.floor(x))
    def c(x):
        return int(np.ceil(x))

    # crop image if image_size < H,W
    if hpad < 0:
        image_stack = image_stack[:, :, -c(hpad) // 2:f(hpad) // 2, :]
        mask = mask[-c(hpad) // 2:f(hpad) // 2, :]
    if wpad < 0:
        image_stack = image_stack[:, :, :, -c(wpad) // 2:f(wpad) // 2]
        mask = mask[:, -c(wpad) // 2:f(wpad) // 2]
    # pad image if image_size > H, W
    if hpad > 0:
        padding = (f(hpad / 2), c(hpad / 2))
        image_stack = np.pad(image_stack, ((0, 0), (0, 0), padding, (0, 0)))
        mask = np.pad(mask, (padding, (0, 0)))
    if wpad > 0:
        padding = (f(wpad / 2), c(wpad / 2))
        image_stack = np.pad(image_stack, ((0, 0), (0, 0), (0, 0), padding))
        mask = np.pad(mask, ((0, 0), padding))
    return image_stack, mask
