import os
import csv
import json
import glob
import argparse


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
    default="/home/tingkeshenlocal/Projects/segment-anything-kevin/out",
    help="Path to either a single input image or folder of images.",
)


def compile_batch(args):
    # files = glob.glob(args.input+"/**/*.json", recursive=True)

    experiment_dirs = [dirpath for dirpath, dirnames, filenames in os.walk(args.input) if not dirnames]

    for d in experiment_dirs:
        os.system('python scripts/compile_segment_counts.py --input {} --output {}.csv'.format(d, d))        


if __name__ == "__main__":
    args = parser.parse_args()
    compile_batch(args)