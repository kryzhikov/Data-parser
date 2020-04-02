# Talking Heads

---



## Step 1:  Creating dataset

---



#### Classes:

1.  [VideoLoader](https://gitlab.com/hvedrung/tlab-talking-heads/-/blob/workflow/VideoLoader.py) - class to load video from youTube via URL returns raw video dir

2.  [VideoTrimmer](https://gitlab.com/hvedrung/tlab-talking-heads/-/blob/workflow/VideoTrimmer.py) - class to trim  raw video to specified intervals with ffmpeg 

3.  [VideoChecker](https://gitlab.com/hvedrung/tlab-talking-heads/-/blob/workflow/VideoChecker.py) - class to check videos to contain only one unique face on each frame

4. [ffe](https://gitlab.com/hvedrung/tlab-talking-heads/-/blob/workflow/ffe.py) - class to work with faces with specified margin and size, vectors from vgg19Face, dlib landmarks etc. 

#### Example:



```python
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')# specify device to run MTCNN on
model = InceptionResnetV1(pretrained='vggface2').eval()#specify embedding model

url = "https://www.youtube.com/watch?v=n7B9utHCUTM"  # url to youtube video

videoLoader = VideoLoader(url) # create youtube loader 
raw_dir = videoLoader.load() # load video to default dir
videoTrimmer = VideoTrimmer(raw_dir, 5) # trim input video with ffmpeg to parts of 5 secs 
videoTrimmer.trim()
print("Trimmed Now Filtering samples!")
check_dir = "./dataset/video/raw/n7B9utHCUTM"  # FIXME path only for this example to check video dir
videoChecker = VideoChecker(check_dir)# filter samples to contain one exact  face
videoChecker.check(device, model)
```


