# lung-cancer-xai

Lung Cancer 


## Data source

Clinical and Radiological data was downloaded from 
[NSCLC-RADIOMICS - The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/collection/nsclc-radiomics/)






## Notebooks
1. 01_Clinical_data_exploration.ipynb : Exploratory data analysis of Clinical Data
2. 02_Data_exploration.ipynb : Exploratory data analysis of Radiology data using ```pydicom``` and ```SimpleITK```
3. 03_Data_preparation.ipynb: Preprocessing : getting torch tensors from DICOM images and preparing dataset for DL
4. 04_Reature_extraction_ResNET.ipynb : Extrating features using ResNET for further processing
5. 05_2DConversion.ipynb : Converting to 2D images using Maximum Intensity Projection (MIP)


## Running Experiments
The scripts for running the experiments are in ```src``` folder.

```bash 
python src/main.py
```