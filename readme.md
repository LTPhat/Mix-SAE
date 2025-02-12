# Mix-SAE
This repository is the implementation of *'Towards Unsupervised Speaker Diarization System for Multilingual Telephone Calls Using Pre-trained Whisper Model and Mixture of Sparse Autoencoders'*.

## Dataset preparation
Prepare your own dataset folder as followed:
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
### 1) Settings
- Create env and install dependencies:

```shell
conda create -n [env_name] python=3.10.15
conda activate [env_name]
```

- Install command-line tool ffmpeg: 

```shell

# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```

- Install dependencies:

```sh
pip install -r requirements.txt
```

### 2) Run 
- Download Whisper model weights (.pt) and put them into ``pre_trained_model/`` folder (``tiny.pt`` available).
- Change model arguments (described in ```run.sh``` file) based on the settings.
- Run script
  
  ```sh
  bash run.sh
  ```