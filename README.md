# It's the public repository of clone of OpenCap system From Standford

## Quick start 
At first check requierd modules
Run the env_test.py

```
python env_test.py
```

provide your own media into media folder and change the path in main.py media_path

```
media_path = "./media/your_media"
```

## Project structure

- **README.md**
- **requirements.txt** 
- **.gitignore**  
- **main.py** — main script 
- **Media** — 
  - *Ready Video*   
  - *Source Video* 
    - *Camera1* 
    - *Camera2* - for video from camera 2
  - *Images* — sliced images
- **Source** — source code  
  - *Models* — models for mediapipe  
  - *Config* — config files  
  - *Scripts* — main scripts like calibration, slice vide for frames and etc.  
  - *Output_data* — output data from Scripts  
  - *Log* — Crash logs  


📂 ProjectX  
├── 📄 README.md  
├── 📄 requirements.txt — environment requirements  
├── 📄 .gitignore  
├── 📄 main.py  
├── 📂 Media — media materials  
│   ├── 📂 Ready Video — video with drawn points on video  
│   ├── 📂 Source Video — video from cameras  
│   │   ├── 📂 Camera1 — video from camera 1  
│   │   └── 📂 Camera2 — video from camera 2  
│   └── 📂 Images — sliced images for calibration  
└── 📂 Source — source code  
    ├── 📂 Models — MediaPipe models  
    ├── 📂 Config — configuration files  
    ├── 📂 Scripts — main scripts (hand_landmark, slicer, calibration and etc.)  
    ├── 📂 Output_data — output data from scripts, data for scripts  
    └── 📂 Log — crash logs *(better to add .gitignore)*


## About models
This project use separeted ready models with extension .task  

run this in console in ./models dir
```
curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```
## Related links
[OpenCap](https://www.opencap.ai/)  
[Simple Stereo | Camera Calibration](https://www.youtube.com/watch?v=hUVyDabn1Mg)  
[OpenCV Python Camera Calibration](https://www.youtube.com/watch?v=H5qbRTikxI4)
