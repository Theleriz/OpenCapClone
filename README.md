# It's the public repository of clone of OpenCap system From Standford

## Quick start 
At first check requierd modules
Run the envTest.py

```
python envTest.py
```

provide your own media into media folder and change the path in main.py media_path

```
media_path = "./media/your_media"
```

## About models
This project use separeted ready models with extension .task  

run this in console in ./models dir
```
curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```
## Related links
[OpenCap](https://www.opencap.ai/)

