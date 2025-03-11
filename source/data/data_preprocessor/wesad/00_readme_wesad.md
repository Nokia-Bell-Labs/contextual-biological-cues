## WESAD

0. Link to the paper: [Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection](https://dl.acm.org/doi/abs/10.1145/3242969.3242985)

1. Specify the path in `source/data/dataset_paths.json` where downloaded and preprocessed data will be stored:
   ```json
   {
     "fatigueset": {
       "raw": "{path_to_save_raw_data}/datasets/fatigueset/raw",
       "processed": "{path_to_save_processed_data}/datasets/fatigueset/processed"
     },
     "wesad": {
        "raw": "{path_to_save_raw_data}/datasets/wesad/raw",
        "processed": "{path_to_save_processed_data}/datasets/wesad/processed"
     }
   }
   ```

2. Create a directory, where the raw data will be downloaded.
   ```
   mkdir -p {path_to_save_raw_data}/datasets/wesad/raw
   ``` 
   
3. Navigate to the `{path_to_save_raw_data}/datasets/wesad/raw` directory.


4. Download the dataset from an official website. The zip file is around `2.1GB`.
   ```
   wget https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download
   ```
5. Unzip the downloaded file using `unzip download` as the download file name is `download`, but it is still a zip file.

6. After unzipping, the directory structure should look like this:
   ``` 
   datasets
   ├── wesad
   │   ├── raw
   │   ├───├── download (original zip file)
   │   │   ├── WESAD (unzipped folder)
   │   │   │   ├── S2 (participant id)
   │   │   │   │   ├── S2.pkl (we use this file to extract data, as it is already aligned)
   │   │   │   │   ├── S2_E4_Data.zip
   │   │   │   │   ├── S2_respiban.txt
   │   │   │   │   ├── S2_quest.csv
   │   │   │   │   ├── S2_readme.txt
   │   │   │   ├── ...(other participants)
   │   │   │   ├── wesad_readme.md
   ```

7. Run `python 01_preprocess_wesad.py` to preprocess the data. As a result, two files will be generated:
   - (1) `wesad_unfiltered_32Hz.csv` - resampled and unfiltered data (no sensor-specific filtering)
   - (2) `wesad_filtered_32Hz.csv` - resampled and filtered data (sensor-specific filtering such as removing noise applied)
   - Both files contain the same data, but the filtered version is cleaner.

8. Run `sh 03_run_feat_engin_wesad.sh` to extract feature-engineered data.