from PIL import Image

def image_grid(imgs, rows, cols):
    w, h = imgs[0].size
    if imgs[0].mode == 'L':
        grid = Image.new('L', size=(cols * w, rows * h))
    else:
        grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid