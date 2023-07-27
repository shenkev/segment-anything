import cv2
import glob
from tqdm import tqdm


def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]

files = glob.glob("/ptmp/tshen/shared/IC9600/images/*")
files = sorted(files)

num_resized = 0

for f in tqdm(files):

    image = cv2.imread(f)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w = image.shape[0:2]
    if (h > 1024) or (w > 1024):
        i = argmax([h, w])
        if i == 0:
            ratio = h/1024
            new_size = [1024, int(w/ratio)]
        else:
            ratio = w/1024
            new_size = [int(h/ratio), 1024]

        small_image = cv2.resize(image, (new_size[1], new_size[0]))
        cv2.imwrite(f, small_image)
        num_resized += 1

print("Done! {} images resized.".format(num_resized))