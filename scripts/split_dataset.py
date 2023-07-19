import os
import glob
import argparse
import shutil


parser = argparse.ArgumentParser(
    description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
        "as well as pycocotools if saving in RLE format."
    )
)

parser.add_argument(
    "--input",
    type=str,
    default="/home/tingkeshenlocal/Pictures/shared/test",
    help="Path to either a single input image or folder of images.",
)


def split_dataset(args):
    GPUS = [0, 1, 2, 3]

    assert os.path.isdir(args.input)

    if not os.path.exists("input"):
        os.makedirs("input")

    files = glob.glob(args.input+"/*")
    n_splits = len(GPUS)

    for i in range(n_splits):
        begin = i*(len(files)//n_splits)
        end = (i+1)*(len(files)//n_splits) if i < n_splits-1 else len(files)
        fs = files[begin:end]

        dest = "input/{}".format(i)
        os.makedirs(dest)

        for f in fs:
            shutil.copy(f, dest)


if __name__ == "__main__":
    args = parser.parse_args()
    split_dataset(args)