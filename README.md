# Segmentation of spinal cord multiple sclerosis lesions

## Robust Spinal Cord MS Lesion Segmentation Across Diverse MRI Protocols and Centers

[![MSJ](https://img.shields.io/badge/MSJ-10.1177/13524585261427333-darkgreen.svg)](https://doi.org/10.1177/13524585261427333)

<img src="https://github.com/user-attachments/assets/6c86548a-0a28-40e4-9d21-219ac310d867" width="500"/>

Official repository for the segmentation of multiple sclerosis (MS) spinal cord (SC) lesions.

This repo contains all the code for training the SC MS lesion segmentation model. The code for training is based on the nnUNetv2 framework. The segmentation model is available as part of [Spinal Cord Toolbox (SCT)](https://spinalcordtoolbox.com/stable/index.html) via the sct_deepseg functionality.

### Citation Information

If you find this work and/or code useful for your research, please cite our paper:

```
@article{doi:10.1177/13524585261427333,
title ={Generalizable spinal cord multiple sclerosis lesion segmentation across MRI contrasts, protocols, and centers},
journal = {Multiple Sclerosis Journal},
volume = {0},
number = {0},
pages = {13524585261427333},
year = {2026},
doi = {10.1177/13524585261427333},
note ={PMID: 42028790},
URL = {https://doi.org/10.1177/13524585261427333},
eprint = {https://doi.org/10.1177/13524585261427333}
author = {Pierre-Louis Benveniste and Laurent Létourneau-Guillon and David Araujo and Lydia Chougar and Dumitru Fetco and Masaaki Hori and Kouhei Kamiya and Steven Messina and Charidimos Tsagkas and Bertrand Audoin and Rohit Bakshi and Elise Bannier and Daniel Blezek and Jean-Christophe Brisset and Virginie Callot and Erik Charlson and Michelle Chen and Olga Ciccarelli and Sarah Demortière and Gilles Edan and Massimo Filippi and Tobias Granberg and Cristina Granziera and Christopher C. Hemond and B. Mark Keegan and Anne Kerbrat and Jan Kirschke and Shannon Kolind and Pierre Labauge and Lisa Eunyoung Lee and Yaou Liu and Caterina Mainero and Julian McGinnis and Nilser Laines Medina and Mark Mühlau and Govind Nair and Kristin P. O’Grady and Jiwon Oh and Russell Ouellette and Alexandre Prat and Daniel S. Reich and Maria A. Rocca and Timothy M. Shepherd and Seth A. Smith and Leszek Stawiarz and Jason Talbott and Roger Tam and Shahamat Tauhid and Anthony Traboulsee and Constantina Andrada Treaba and Paola Valsasina and Zachary Vavasour and Marios Yiannakas and Hervé Lombaert and Julien Cohen-Adad}
}
```

### How to use the model

Install the Spinal Cord Toolbox (SCT) [here](https://spinalcordtoolbox.com/stable/user_section/installation.html).

Run the command: 
```console
sct_deepseg lesion_ms
```

More details can be found in the user section [here](https://spinalcordtoolbox.com/stable/user_section/command-line/sct_deepseg.html#sct-deepseg)

<details>
<summary><b>Code description</b></summary>

The repository contains all the code for the SC MS lesion segmentation project:
- `compute_canada_scripts`: code used to train the model on compute canada
- `dataset_aggregation`: code used to aggregate all the data
- `dataset_analysis`: code used to analyze the data
- `evaluation`: code used to evaluate the performance of the model
- `nnunet`: code used for training with nnunet
- `post-processing`: code used for post-processing
</details>

## Longitudinal tracking of multiple sclerosis lesions in the spinal cord: A validation study

TODO: if accepted add icon here

<img src="https://github.com/user-attachments/assets/cdb12ab9-aa16-42b9-b0f2-8fcfbf2e8062" width="700"/>


Official repository of the longitudinal lesion tracking of MS SC lesions.

This repo contains the code for evaluating 5 strategies to track SC MS lesions across timepoints. More details can be find in the paper.

```
TODO: if accepted add the citation here
```

All the code is contained in the folder [longitudinal_tracking](./longitudinal_tracking). 