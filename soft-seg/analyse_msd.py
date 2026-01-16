import json

input_msd = "/home/plbenveniste/net/soft-seg/downsample_exp/input/msd_data.json"

# Open the msd file
with open(input_msd, 'r') as f:
    msd_data = json.load(f)
images = msd_data['test']

contrast_count_dict = {}
for img in images:
    img_contrast = img['contrast']
    if img_contrast not in contrast_count_dict:
        contrast_count_dict[img_contrast] = 0
    contrast_count_dict[img_contrast] += 1

print(contrast_count_dict)