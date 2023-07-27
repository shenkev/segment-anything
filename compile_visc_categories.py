import os
import glob
import json


folder = "/ptmp/tshen/shared/VISCHEMA_SUN"

dirs = [dirpath for dirpath, dirnames, filenames in os.walk(folder) if not dirnames]

files_to_cat = {}

for d in dirs:
    files = glob.glob(d+"/*.jpg")

    for f in files:
        files_to_cat[os.path.basename(f)] = os.path.basename(d)

with open(folder + "/file_categories.json", 'w') as f:
    json.dump(files_to_cat, f)

print("done")