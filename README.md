BioQ: Towards Context-Aware Multi-Device Collaboration with Bio-cues (ACM SenSys '25)
---
This is official PyTorch implementation of "BioQ: Towards Context-Aware Multi-Device Collaboration with Bio-cues" (ACM SenSys '25) by [Adiba Orzikulova](https://adibaorz.super.site), [Diana A. Vasile](https://www.linkedin.com/in/dvasile7/?originalSubdomain=uk), [Chi Ian Tang](https://iantang.co), [Fahim Kawsar](https://www.fahim-kawsar.net), [Sung-Ju Lee](https://sites.google.com/site/wewantsj/), and [Chulhong Min](http://chulhongmin.com). 

### Tested Environment

- Ubuntu 22.04
- NVIDIA GeForce RTX 3090 GPUs


### Step-by-Step Guide

1. Create a new conda environment with required dependencies.
    ```
    conda env create -f bioq.yml
    ```
   
2. Download and preprocess datasets. 
   To preprocess datasets, refer to the dataset-specific instructions in `00_readme_{dataset_name}.md`, located in `source/data/data_preprocessor/{dataset_name}`. 


3. Run experiments. There are two tasks in the paper: 
    - Task 1: Cue (embedding) generation
    - Task 2: Cue (embedding) matching
    
   First, run the cue generation task: 
   ```
   sh scripts/embgen/{method_name}/embgen_{method_name}_{dataset_name}.sh
  
    # e.g., to run BioQ with wesad dataset:
    sh scripts/embgen/bioq/embgen_bioq_wesad.sh
   ```
    
    Then, run the cue matching task:
    ```
    sh scripts/matching/{method_name}/matching_{method_name}_{dataset_name}.sh
      
    # e.g., to run BioQ with wesad dataset:
    sh scripts/matching/bioq/matching_bioq_wesad.sh
    ```
   
    
### Citation
```
@inproceedings{orzikulova2025bioq,
  title={BioQ: Towards Context-Aware Multi-Device Collaboration with Bio-cues},
  author={Orzikulova, Adiba and Vasile, Diana A. and Tang, Chi Ian and Kawsar, Fahim and Lee, Sung-Ju and Min, Chulhong},
  booktitle={Proceedings of the 23rd ACM Conference on Embedded Networked Sensor Systems},
  year={2025}
}
```
----