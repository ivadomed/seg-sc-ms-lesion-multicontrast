"""
This script takes as input longitudinal images and the corresponding lesion segmentations and computes a report of lesion changes over time.

Input: 
    -i: folder containing longitudinal images
    -l: folder containing corresponding lesion segmentations
    -sc: folder containing corresponding spinal cord segmentations
    -o: output folder to save the report

Author: Pierre-Louis Benveniste
"""
import os
import argparse
import numpy as np
from pathlib import Path
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from datetime import datetime
from scipy.ndimage import label


def parse_arguments():
    parser = argparse.ArgumentParser(description="Compute lesion change report from longitudinal images and segmentations.")
    parser.add_argument('-i', '--images', required=True, help='Folder containing longitudinal images')
    parser.add_argument('-l', '--lesion-seg', required=True, help='Folder containing corresponding lesion segmentations')
    parser.add_argument('-s', '--sc-seg', required=True, help='Folder containing corresponding spinal cord segmentations')
    parser.add_argument('-o', '--output', required=True, help='Output folder to save the report')
    return parser.parse_args()


def compute_lesion_volume(seg_file):
    """
    Compute total lesion volume from a segmentation file.
    In this function, we assume that the segmentation file is a binary mask where lesions are marked with 1s.
    """
    # Load the lesion mask
    lesion_img = nib.load(seg_file)
    lesion_data = lesion_img.get_fdata()

    # Get voxel dimensions in mm
    voxel_dims = lesion_img.header.get_zooms()

    # Calculate the volume of a single voxel in cubic millimeters
    voxel_volume = np.prod(voxel_dims)

    # Compute volume in case of soft segmentation, i.e. volume of a voxel labeled 0.5 is counted as half a voxel
    seg_data = lesion_data[lesion_data > 1e-6]
    lesion_voxel_volume = np.sum(seg_data)

    # Calculate the total lesion volume
    lesion_volume = lesion_voxel_volume * voxel_volume
    return lesion_volume


def count_lesions(seg_file):
    """
    Count the number of distinct lesions in the segmentation file using connected component analysis.
    """
    # Load the lesion mask
    lesion = nib.load(seg_file)
    lesion_data = lesion.get_fdata()

    # Label connected components (with connectivity=1 for 26-connectivity in 3D)
    labeled_array, num_features = label(lesion_data, structure=np.ones((3,3,3)))

    return num_features


def screenshot_seg(seg_file, img_file, img_date, output_folder):
    """
    Create a screenshot of the middle slice of the segmentation in the sagittal plane.

    Input:
        - seg_file: path to the segmentation file
        - img_file: path to the anatomical image file
        - img_date: imaging date string for naming the output file
        - output_folder: folder to save the screenshot
    Output:
        - screenshot_file_path: path to the saved screenshot file
    """
    # Load the lesion mask
    lesion = nib.load(seg_file)
    lesion_data = lesion.get_fdata()
    # Load the image
    img = nib.load(img_file)
    img_data = img.get_fdata()

    # Get the center of gravity of the lesion mask
    coords = np.argwhere(lesion_data > 0)
    cog = np.mean(coords, axis=0).astype(int)

    if img_date != '20160429':
        cog += 1  # small hack to adjust for misalignment in this specific case

    # Get orientation of the image
    orientation = nib.aff2axcodes(lesion.affine)

    # Assert that both img and seg have the same orientation
    img_orientation = nib.aff2axcodes(img.affine)
    assert orientation == img_orientation, "Image and segmentation have different orientations!"

    # Get the RL axis
    rl_axis = orientation.index('R') if 'R' in orientation else orientation.index('L')
    # Extract the sagittal slice at the center of gravity
    if rl_axis == 0:
        l_slice = lesion_data[cog[0], :, :]
        i_slice = img_data[cog[0], :, :]
    elif rl_axis == 1:
        l_slice = lesion_data[:, cog[1], :]
        i_slice = img_data[:, cog[1], :]
    else:
        l_slice = lesion_data[:, :, cog[2]]
        i_slice = img_data[:, :, cog[2]]

    # Set the orientation to have superior at the top and anterior to the left
    if rl_axis == 0:
        l_slice = np.rot90(l_slice)
        i_slice = np.rot90(i_slice)
    elif rl_axis == 1:
        l_slice = np.flipud(np.rot90(l_slice))
        i_slice = np.flipud(np.rot90(i_slice))
    else:
        l_slice = np.fliplr(np.rot90(l_slice))
        i_slice = np.fliplr(np.rot90(i_slice))

    # 4. Process Anatomical Slice (Grayscale -> RGB)
    # Normalize to 0-255 based on robust intensity (99th percentile)
    vmin, vmax = i_slice.min(), np.percentile(i_slice, 99)
    a_norm = np.clip((i_slice - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
    # Convert to PIL RGB
    anat_pil = Image.fromarray(a_norm).convert("RGB")
    # 3. USE PIL ENHANCERS (Optional extra "oomph")
    # Increase Contrast (1.0 is original, 1.2 is 20% more)
    contrast_enhancer = ImageEnhance.Contrast(anat_pil)
    anat_pil = contrast_enhancer.enhance(1.4)
    # Increase Brightness/Exposure
    bright_enhancer = ImageEnhance.Brightness(anat_pil)
    anat_pil = bright_enhancer.enhance(1.5)

    # 5. Process Segmentation Slice (Binary -> Red RGB)
    # Create an empty RGB array
    seg_rgb = np.zeros(l_slice.shape + (3,), dtype=np.uint8)
    # Assign Red: [255, 0, 0] where mask is > 0
    seg_rgb[l_slice > 0] = [255, 0, 0]
    seg_pil = Image.fromarray(seg_rgb).convert("RGB")

    # 6. BLEND (The Secret Sauce)
    # alpha=0.0 is pure anatomy, alpha=1.0 is pure red mask. 
    # 0.3-0.4 is usually sweet for seeing anatomy through the red.
    blended_img = Image.blend(anat_pil, seg_pil, alpha=0.3)
    # Save the blended image
    output_path = os.path.join(output_folder, f"refined_overlay_{img_date}.png")
    blended_img.save(output_path)

    return output_path
    

def generate_html_report(df, screenshots, output_folder):
    """
    Generates a clean HTML report with screenshots side-by-side and the volume graph.
    """
    # Create HTML table rows for images
    # We sort screenshots by date to match the side-by-side requirement
    sorted_shots = sorted(screenshots, key=lambda x: x['imaging_date'])
    
    img_cells = ""
    for shot in sorted_shots:
        # Use relative path for the HTML to work locally
        rel_path = os.path.basename(shot['screenshot_path'])
        img_cells += f"""
            <div style="flex: 1; text-align: center; padding: 10px;">
                <h3>Date: {shot['imaging_date']}</h3>
                <img src="{rel_path}" style="width: 100%; max-width: 400px; border-radius: 8px; border: 1px solid #ddd;">
            </div>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Lesion Evolution Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f4f4f9; }}
            .container {{ max-width: 1200px; margin: auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .header {{ text-align: center; border-bottom: 2px solid #333; padding-bottom: 10px; }}
            .row {{ display: flex; flex-direction: row; justify-content: space-around; align-items: flex-start; margin-top: 20px; }}
            .graph-container {{ text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; }}
            .graph-container img {{ width: 80%; border: 1px solid #ccc; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Longitudinal Lesion Analysis Report</h1>
            </div>
            
            <div class="row">
                {img_cells}
            </div>

            <div class="graph-container">
                <h2>Total lesion volume across time</h2>
                <img src="lesion_volume_over_time.png">
            </div>
            <div class="graph-container">
                <h2>Number of lesions across time</h2>
                <img src="num_lesions_over_time.png">
            </div>
        </div>
    </body>
    </html>
    """

    report_path = os.path.join(output_folder, "report.html")
    with open(report_path, "w") as f:
        f.write(html_content)
    
    print(f"Report generated at: {report_path}")
    return report_path


def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Build the output folder
    os.makedirs(args.output, exist_ok=True)

    # Get all images
    images = list(Path(args.images).glob('*.nii.gz'))
    images = [str(img) for img in images if "flatten" not in str(img)]  # Exclude already flattened images  
    # Find corresponding segmentation files
    lesion_segs = []
    sc_segs = []
    for img in images:
        lesion_seg_file = os.path.join(args.lesion_seg, str(img).split("/")[-1].replace('.nii.gz', '_lesion-seg.nii.gz'))
        if os.path.exists(lesion_seg_file):
            lesion_segs.append(lesion_seg_file)
        else:
            raise FileNotFoundError(f"Segmentation file not found {lesion_seg_file}")
        sc_seg_file = os.path.join(args.sc_seg, str(img).split("/")[-1].replace('.nii.gz', '_sc-seg.nii.gz'))
        if os.path.exists(sc_seg_file):
            sc_segs.append(sc_seg_file)
        else:
            raise FileNotFoundError(f"Spinal cord segmentation file not found {sc_seg_file}")
        
    ## Now we improve the sagittal view by making sure the SC is flattened on the IS axis
    for img, lesion_seg, sc_seg in zip(images, lesion_segs, sc_segs):
        # Flatten the image
        assert os.system(f"sct_flatten_sagittal -i {img} -s {sc_seg}")==0, f"Error flattening image {img}"
        # We convert the lesion segmentations to float first to avoid interpolation issues
        lesion_seg_float = lesion_seg.replace('.nii.gz', '_float.nii.gz')
        assert os.system(f"sct_image -i {lesion_seg} -o {lesion_seg_float} -type float32")==0, f"Error converting lesion segmentation {lesion_seg} to float"
        lesion_data = nib.load(lesion_seg_float).get_fdata()
        print(lesion_data.dtype)
        # Flatten the lesion seg
        assert os.system(f"sct_flatten_sagittal -i {lesion_seg_float} -s {sc_seg}")==0, f"Error flattening lesion segmentation {lesion_seg_float}"
        # Then we need to binarize the lesion segmentation again (threshold at 0.5)
        assert os.system(f"sct_maths -i {lesion_seg_float.replace('.nii.gz', '_flatten.nii.gz')} -o {lesion_seg_float.replace('.nii.gz', '_flatten.nii.gz')} -bin 0.5")==0, f"Error binarizing lesion segmentation {lesion_seg_float.replace('.nii.gz', '_flatten.nii.gz')}"
        
    # Update the images and segmentations lists to point to the flattened versions
    images = [str(img).replace('.nii.gz', '_flatten.nii.gz') for img in images]
    lesion_segs = [str(seg).replace('.nii.gz', '_float_flatten.nii.gz') for seg in lesion_segs]

    lesion_changes_df = []
    path_screenshots = []

    for seg_file, img in zip(lesion_segs, images):
        # Compute lesion total volume
        lesion_total_volume = compute_lesion_volume(seg_file)
        # Number of lesions
        num_lesions = count_lesions(seg_file)

        # Extract imaging date
        img_date = seg_file.split("/")[-1].split("_")[1].replace('ses-', '')

        lesion_changes_df.append({
            'image_file': str(img),
            'segmentation_file': seg_file,
            'imaging_date': img_date,
            'lesion_total_volume': lesion_total_volume,
            'num_lesions': num_lesions
        })

        # For each timepoint we also want to have a screen shot of the middle slice (of the segmentation) in the sagittal plane
        screenshot_file_path = screenshot_seg(seg_file, img, img_date, args.output)
        img_date_formated = datetime.strptime(img_date, '%Y%m%d').strftime('%Y-%m-%d')
        path_screenshots.append({'imaging_date': img_date_formated, 'screenshot_path': screenshot_file_path})
    
    # Convert to DataFrame
    lesion_changes_df = pd.DataFrame(lesion_changes_df)
    # Format imaging_date as datetime
    lesion_changes_df['imaging_date'] = pd.to_datetime(lesion_changes_df['imaging_date'], format='%Y%m%d')
    # Sort by imaging date
    lesion_changes_df = lesion_changes_df.sort_values(by='imaging_date')
    
    # Plot a graph of the lesion volume over time
    plt.figure(figsize=(10, 6))
    plt.plot(lesion_changes_df['imaging_date'], lesion_changes_df['lesion_total_volume'], marker='o')
    plt.title('Total lesion volume across time')
    plt.xlabel('Imaging Date')
    plt.ylabel('Total Lesion Volume (mm³)')
    # Max and min y limit with 20% margin
    plt.ylim(bottom=lesion_changes_df['lesion_total_volume'].min() * 0.8, top=lesion_changes_df['lesion_total_volume'].max() * 1.2)
    plt.grid()
    plt.savefig(os.path.join(args.output, 'lesion_volume_over_time.png'))
    plt.close()

    # Plot a graph of the number of lesions over time
    plt.figure(figsize=(10, 6))
    plt.plot(lesion_changes_df['imaging_date'], lesion_changes_df['num_lesions'], marker='o', color='orange')
    plt.title('Number of lesions across time')
    plt.xlabel('Imaging Date')
    plt.ylabel('Number of Lesions')
    # Max and min y limit with 20% margin
    plt.ylim(bottom=lesion_changes_df['num_lesions'].min() - 1, top=lesion_changes_df['num_lesions'].max() + 1)
    # Grid must be intervals of 1
    plt.yticks(range(int(lesion_changes_df['num_lesions'].min()) - 1, int(lesion_changes_df['num_lesions'].max()) + 2))
    plt.grid()
    plt.savefig(os.path.join(args.output, 'num_lesions_over_time.png'))
    plt.close()

    # Generate HTML report
    generate_html_report(lesion_changes_df, path_screenshots, args.output)





if __name__ == "__main__":
    main()