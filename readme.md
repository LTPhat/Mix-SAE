# Mix-SAE
This repository is the implementation of *'Towards Unsupervised Speaker Diarization System for Multilingual Telephone Calls Using Pre-trained Whisper Model and Mixture of Sparse Autoencoders'*.

## Dataset preparation
Prepare dataset folder as followed:
```                  
      ├── dataset                   
      │  ├── english 
      |  |      ├── samples
      |  |      |   ├── sample1.wav
      |  |      |   ├── sample2.wav
      |  |      |   ├── ...
      |  |      ├── label
      |  |      |   ├── sample1.rttm
      |  |      |   ├── sample2.rttm
      │  ├── french
      |  |      ├── samples
      |  |      |   ├── sample1.wav
      |  |      |   ├── sample2.wav
      |  |      |   ├── ...
      |  |      ├── label
      |  |      |   ├── sample1.rttm
      |  |      |   ├── sample2.rttm
      |  |      |   ├── ...
      │  ├── other_language
      |  |      ├── ...
```
## Run code
### 1) Create env and install dependencies
```shell
conda create -n [env_name] python=3.10.15
conda activate [env_name]
pip install -r requirements.txt
```
### 2) Run 
- Download Whisper model weights (.pt) and put them into ``pre_trained_model/`` folder (``tiny.pt`` available).
- Change arguments (described in ```run.sh``` file) based on the settings.
- Run script
  
  ```sh
  bash run.sh
  ```