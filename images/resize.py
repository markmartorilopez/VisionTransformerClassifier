import glob2 as glob
import PIL
from PIL import Image

def resizeImages(folder,image_size):
    for img in glob.iglob(folder+'/*.png', recursive=True):
        image = Image.open(img)
        print(f"old image size = {image.size}")
        new_image = image.resize((image_size, image_size))
        new_image.save(img)
        image = Image.open(new_image)
        print(F"new image size = {image.size}")
    return 'Images resized'