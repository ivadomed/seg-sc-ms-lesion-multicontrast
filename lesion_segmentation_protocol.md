# Lesion segmentation protocol

The following details the protocol for Multiple Sclerosis (MS) lesion segmentation in the spinal cord.
Imaging the spinal cord is often essential to confirm the diagnosis of MS. That is because the lesions of the spinal cord are included in the McDonald diagnostic criteria, which considers dissemination in space and in time [(Thompson et al. 2018)](https://pubmed.ncbi.nlm.nih.gov/29275977/). While the MAGNIMS-CMSC-NAIMS working group recommends to use at least two sagittal images for MS diagnosis, still, axial imaging is mentioned as optional in international imaging guidelines [(Wattjes et al. 2021)](https://pubmed.ncbi.nlm.nih.gov/34139157/).
For detecting MS lesions in the spinal cord, two main contrasts emerge: PSIR and STIR contrasts. New studies [(Peters et al. 2024)](https://pubmed.ncbi.nlm.nih.gov/38289376/)[(Fechner et al. 2019)](https://pubmed.ncbi.nlm.nih.gov/30679225/) showed that using PSIR contrasts improved MS lesion detection in the spinal cord. [(Fechner et al. 2019)](https://pubmed.ncbi.nlm.nih.gov/30679225/) showed that the PSIR contrast showed a higher signal-to-noise (SNR) ratio compared to the STIR contrast. 

## Criteria to segment MS lesions in the spinal cord
Based on the definition proposed for MS lesions in the spinal cord in the 2017 revisions of the McDonald criteria:
"A hyperintense lesion in the cervical, thoracic, or lumbar spinal cord seen on T2 plus short tau inversion recovery, proton-density images, or other appropriate sequences, or in two planes on T2 images."  [(Thompson et al. 2018)](https://pubmed.ncbi.nlm.nih.gov/29275977/) [(Filippi et al. 2016)](https://pubmed.ncbi.nlm.nih.gov/26822746/) [(Brownlee et al. 2018)](https://pubmed.ncbi.nlm.nih.gov/27889190/) [(Rovira et al. 2016)](https://pubmed.ncbi.nlm.nih.gov/26149978/) 

- Do not segment lesions in images with too many artifacts (such as this [example](https://github.com/ivadomed/canproco/issues/53#issue-1938136790)). Preferably, add the image to the exclude file so that it isn’t used for model training…
- When segmenting lesions on thick slices, always look at the adjacent slices, as partial volume effect can sometimes reduce the appearance of a lesion (close to noise level).
- Unless otherwise stated, do not segment lesions above the first vertebrae (because here we focus only on MS lesions in the spinal cord). 
- If you have any doubt and/or are not 100% confident about one (or more) lesion(s) segmentation(s), flag the subject and report it for external validation of the segmentation.

## How to manually segment lesions

- MS spinal cord lesions can either (i) be automatically segmented from an algorithm and then manually corrected, or (ii) manually segmented from scratch. In the former case, make sure to use the JSON file that was created by the automatic segmentation algorithm, in order to track provenance:

```json
{
        "GeneratedBy": [
            {
                "Name": "2D nnUNet model model_ms_seg_sc-lesion_regionBased.zip",
                "Version": "https://github.com/ivadomed/canproco/releases/tag/r20240125",
                "Date": "2024-01-26"
            }
        ]
    }
```

- To manually create/correct the segmentation, please use the manual-correction (https://github.com/spinalcordtoolbox/manual-correction) repository. The command can be inspired by this: 

```console
python manual_correction.py -path-img ~/data/canproco -config ~/config_seg.yml  -path-label ~/data/canproco/derivatives/labels  -suffix-files-lesion _lesion-manual -fsleyes-dr="-40,70"
``` 

- A Quality Control (QC) report should be produced using SCT, and added to a GitHub issue for further validation by other investigators. Using SCT, you can review lesion segmentation in the axial or sagittal plane:

```console
sct_qc -i {image_file} -d {lesion_seg_file} -s {sc_seg_file} -p sct_deepseg_lesion -plane {sagittal, axial} -qc {canproco_qc_folder}
```  

- If you are not sure about the segmentation on a subject, it should be flagged on GitHub for a more open discussion: here are some examples [(1)](https://github.com/ivadomed/ms-lesion-agnostic/issues/4#issuecomment-1947326493) and [(2)](https://github.com/ivadomed/ms-lesion-agnostic/issues/4#issuecomment-1947338624)

## Step 1: Software to create/correct lesions
It is common practice to use FSLeyes at NeuroPoly for visual inspection of MRI images and manual segmentation of MS lesions. Therefore, naturally, the first step of the lesion segmentation process is to complete the FSLeyes tutorial ([FSLeyes documentation](https://open.win.ox.ac.uk/pages/fsl/fsleyes/fsleyes/userdoc/) and [video tutorial](https://www.youtube.com/playlist?list=PLIQIswOrUH69qFMNg8KYkEGkvCNEwlnfT)). Trainees are encouraged to learn keyboard shortcuts (ctrl+F to toggle an image, shift+↑ to scroll through volumes, ...).

Furthermore, it is recommended to get familiar with SCT for creating QCs and for manual correction ([SCT tutorial](https://spinalcordtoolbox.com/user_section/tutorials.html)). 

## Step 2: Spinal cord anatomy and lesion segmentation
Before, moving on to MS lesion segmentation, trainees are advised to study the neuroanatomical structures of healthy spinal cords. Trainees should look at healthy spinal cords in MRI images of different contrasts: T2w, T1w, PSIR, STIR, MP2RAGE... A public dataset will be built for this purpose.

To learn the specificity of MS lesions, trainees should work on differentiating MS lesions from other diseases. 

One of the most challenging tasks of MS lesion segmentation is to distinguish the border of a lesion and the cerebrospinal fluid (CSF). To learn where to draw the lesion border, a set of tricky examples validated by a NeuroRadioligist will be created. 

Finally, for trainees will little or no experience with MS lesion segmentation, a checklist will be built to avoid being overwhelmed by the multiple images/contrasts/acquisitions. We typically recommend starting with the view in the highest resolution (often the sagittal view) to first identify lesions and to move to other contrast/acquisition to validate the segmentation borders and lesion detection. During this step, playing with the brightness and the contrast is key. After locating the lesion to be traced, we recommend starting in a middle slice around the middle of the lesion and then moving toward each end of the lesioned area. We also recommended frequently scrolling back and forth around the slice they are tracing on to ensure border consistency.

## Step 3: Manual segmentation assessment
After manual segmentation of MS SC lesion in 5 cases, trainees will receive feedback (from a NeuroRadiologist or a comparison with a QC with the real segmentation). One week later, they will be asked to re-segment the same images as well as 2 other images without using their previous segmentation to validate their improvements. The segmentation should also be accompanied by a JSON file for the data to be BIDS compliant. 

## Taxonomy to evaluate lesion segmentation
The following section details the different types of errors which occur during lesion segmentation. It is based on the condensed Nascimento Taxonomy:

<img width="522" alt="nascimento_taxonomy" src="https://github.com/ivadomed/canproco/assets/67429280/36d9e45e-4a36-40f0-a4f5-e5f3ea3f06a0">

## Sources
This lesion segmentation protocol was inspired by these resources: 
- deSouza NM, van der Lugt A, Deroose CM, et al. Standardised lesion segmentation for imaging biomarker quantitation: a consensus recommendation from ESR and EORTC. Insights Imaging. 2022;13(1):159. Published 2022 Oct 4. doi:10.1186/s13244-022-01287-4 : [link](https://pubmed.ncbi.nlm.nih.gov/36194301/)
- Lo BP, Donnelly MR, Barisano G, Liew SL. A standardized protocol for manually segmenting stroke lesions on high-resolution T1-weighted MR images. Front Neuroimaging. 2023;1:1098604. Published 2023 Jan 10. doi:10.3389/fnimg.2022.1098604 : [link](https://pubmed.ncbi.nlm.nih.gov/37555152/)

