# Segmentation of spinal cord multiple sclerosis lesions

## Generalizable spinal cord multiple sclerosis lesion segmentation across MRI contrasts, protocols, and centers

[![MSJ](https://img.shields.io/badge/MSJ-10.1177/13524585261427333-darkgreen.svg)](https://doi.org/10.1177/13524585261427333)

<p float="left">
    <img src="https://github.com/user-attachments/assets/6c86548a-0a28-40e4-9d21-219ac310d867" width="500"/>
    <img src="https://github.com/user-attachments/assets/d78e9296-d7dc-4217-9e03-eae8d8ddebd1" width="500"/>
</p>


Official repository for the segmentation of multiple sclerosis (MS) spinal cord (SC) lesions.

This repo contains all the code for training the SC MS lesion segmentation model. The code for training is based on the nnUNetv2 framework. The segmentation model is available as part of [Spinal Cord Toolbox (SCT)](https://spinalcordtoolbox.com/stable/index.html) via the `sct_deepseg lesion_ms` command (more details [here](https://spinalcordtoolbox.com/stable/user_section/command-line/deepseg/lesion_ms.html)).

<details>
<summary><b>Description of the code in this repo</b></summary>

The repository contains all the code for the SC MS lesion segmentation project:
- `compute_canada_scripts`: code used to train the model on compute canada
- `dataset_aggregation`: code used to aggregate all the data
- `dataset_analysis`: code used to analyze the data
- `evaluation`: code used to evaluate the performance of the model
- `nnunet`: code used for training with nnunet
- `post-processing`: code used for post-processing
</details>

### Citation Information

If you find this work and/or code useful for your research, please cite our paper:

```
@article{doi:10.1177/13524585261427333,
title ={Generalizable spinal cord multiple sclerosis lesion segmentation across MRI contrasts, protocols, and centers},
journal = {Multiple Sclerosis Journal},
volume = {32},
number = {6},
pages = {598-613},
year = {2026},
doi = {10.1177/13524585261427333},
note ={PMID: 42028790},
URL = {https://doi.org/10.1177/13524585261427333},
eprint = {https://doi.org/10.1177/13524585261427333},
author = {Pierre-Louis Benveniste and Laurent Létourneau-Guillon and David Araujo and Lydia Chougar and Dumitru Fetco and Masaaki Hori and Kouhei Kamiya and Steven Messina and Charidimos Tsagkas and Bertrand Audoin and Rohit Bakshi and Elise Bannier and Daniel Blezek and Jean-Christophe Brisset and Virginie Callot and Erik Charlson and Michelle Chen and Olga Ciccarelli and Sarah Demortière and Gilles Edan and Massimo Filippi and Tobias Granberg and Cristina Granziera and Christopher C. Hemond and B. Mark Keegan and Anne Kerbrat and Jan Kirschke and Shannon Kolind and Pierre Labauge and Lisa Eunyoung Lee and Yaou Liu and Caterina Mainero and Julian McGinnis and Nilser Laines Medina and Mark Mühlau and Govind Nair and Kristin P. O’Grady and Jiwon Oh and Russell Ouellette and Alexandre Prat and Daniel S. Reich and Maria A. Rocca and Timothy M. Shepherd and Seth A. Smith and Leszek Stawiarz and Jason Talbott and Roger Tam and Shahamat Tauhid and Anthony Traboulsee and Constantina Andrada Treaba and Paola Valsasina and Zachary Vavasour and Marios Yiannakas and Hervé Lombaert and Julien Cohen-Adad},
}
```