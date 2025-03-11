## FatigueSet

0. Link to the paper: [FatigueSet: A Multi-modal Dataset for Modeling Mental Fatigue and Fatigability](https://link.springer.com/chapter/10.1007/978-3-030-99194-4_14)

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
   mkdir -p {path_to_save_raw_data}/datasets/fatigueset/raw
   ``` 
   
3. Navigate to the `{path_to_save_raw_data}/datasets/fatigueset/raw` directory.

4. Download the dataset from an official website. The zip file is around `612MB`.
   ```
   wget https://sensix.tech/datasets/fatigueset/fatigueset.zip
   ```
5. Unzip the downloaded file. Unzipped file is `3.2GB`. Unzipping will create another directory called `fatigueset` inside the `raw` directory.
6. After unzipping, the directory structure should look like this:
   ``` 
   datasets
   ├── fatigueset
   │   ├── raw
   │   ├───├── fatigueset.zip
   │   │   ├── fatigueset
   │   │   │   ├── 01 (participant id)
   │   │   │   │   ├── 01 (session id)
   │   │   │   │   │   ├── ear_acc_left.csv
   │   │   │   │   │   ├── ear_acc_right.csv
   │   │   │   │   │   ├── ear_gyro_left.csv
   │   │   │   │   │   ├── ear_gyro_right.csv
   │   │   │   │   │   ├── ear_ppg_left.csv
   │   │   │   │   │   ├── ear_ppg_right.csv
   │   │   │   │   │   ├── ...
   │   │   │   │   ├── ... (other sessions)
   │   │   │   ├── ...(other participants)
   │   │   │   ├── README.md
   │   │   │   ├── metadata.csv
   │   │   │   ├── other files (not used)
   ```

7. Navigate to the `source/data/data_preprocessor/fatigueset` directory.

8. Run `python 01_preprocess_fatigueset.py` to preprocess the data. As a result, two files will be generated:
   - (1) `fatigueset_unfiltered_100Hz.csv` - resampled and unfiltered data (no sensor-specific filtering)
   - (2) `fatigueset_filtered_100Hz.csv` - resampled and filtered data (sensor-specific filtering such as removing noise applied)
   - Both files contain the same data, but the filtered version is cleaner.

9. Run `sh 03_run_feat_engin_fatigueset.sh` to extract feature-engineered data.
