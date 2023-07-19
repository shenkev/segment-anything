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
    default="/home/tingkeshenlocal/Projects/segment-anything-kevin/out/test2",
    help="Path to either a single input image or folder of images.",
)

parser.add_argument(
    "--output",
    type=str,
    default="/home/tingkeshenlocal/Projects/segment-anything-kevin/out/test2_nsegments.csv",
    help="Path to either a single input image or folder of images.",
)

def compile(args):
    files = glob.glob(args.input+"/**/*.json", recursive=True)

    fname_to_nsegments = {}

    for fpath in files:
        with open(fpath, "r") as f:
            masks = json.load(f)
            _, fname = os.path.split(fpath)

            fname_to_nsegments[fname] = len(masks)

    with open(args.output, 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in fname_to_nsegments.items():
            writer.writerow([key, value])
    

if __name__ == "__main__":
    args = parser.parse_args()
    compile(args)