import numpy as np
from PIL import Image
import os

data_dir = "/path/to/conceptual_captions/validation"
output_dir = "/path/to/conceptual_caption/fid-validation"

for img_name in os.listdir(os.path.join(data_dir)):
    img = Image.open(os.path.join(data_dir, img_name))
    img = img.resize((256, 256))
    img = np.array(img)
    np.save(os.path.join(output_dir, img_name + ".npy"), img)
