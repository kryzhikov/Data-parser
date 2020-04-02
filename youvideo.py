import pafy
from utils import *
from tqdm import tqdm
import torch
import os
from PIL import Image
import cv2


class YouVideo(object):
    def __init__(self, url, rawAudioDir="./dataset/audio/raw/", rawVideoDir="./dataset/video/raw/",
                 processedFramesPath="./dataset/video/processed/"):
        """
            :param url: url to youtube video
            :param rawAudioDir: directory to save raw .m4a file
            :param rawVideoDir: directory to save raw .mp4 file
            :param processedFramesPath: directory to save processed frames
            Creates class of YouTube video
        """
        self.url = url
        self.youtubeObj = pafy.new(url)
        self.rawVideoDir = rawVideoDir
        self.rawAudioDir = rawAudioDir
        for i in self.youtubeObj.audiostreams:
            if i.extension == "m4a":
                print(i)
                self.rawAudio = i
                break
        self.ID = self.youtubeObj.videoid
        self.rawAudioPath = rawAudioDir
        if not os.path.exists(rawAudioDir):
            os.makedirs(rawAudioDir)
        self.rawAudio.download(filepath=self.rawAudioDir)
        self.rawVideo = self.youtubeObj.getbest(preftype="mp4")
        self.rawVideoDir = rawVideoDir
        if not os.path.exists(rawVideoDir):
            os.makedirs(rawVideoDir)
        self.rawVideo.download(filepath=self.rawVideoDir)
        self.processedFramesPath = processedFramesPath
        if not os.path.exists(processedFramesPath):
            os.makedirs(processedFramesPath)
        audioName = [
            x for x in os.listdir(rawAudioDir)
            if os.path.isfile(os.path.join(rawAudioDir, x))
        ][0]
        os.rename(rawAudioDir + audioName, rawAudioDir + self.ID + ".m4a")
        videoName = [
            x for x in os.listdir(rawVideoDir)
            if os.path.isfile(os.path.join(rawVideoDir, x))
        ][0]
        os.rename(rawVideoDir + videoName, rawVideoDir + self.ID + ".mp4")
        self.rawAudioPath = rawAudioDir + self.ID + ".m4a"
        self.rawVideoPath = rawVideoDir + self.ID + ".mp4"

    def extract_face(self, device=None, size=512, margin=0):
        """
            :param device: device to ran MTCNN on
            :param size: size of extracted images size x size
            :param margin: margin of extracted images
            extracts frames from video with cropped face

            returns list of error extractions (2 faces or no faces)

        """

        device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu') if device is None else device
        cap = cv2.VideoCapture(self.rawVideoPath)
        i = 0
        error_frames = []
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=length + 1)
        while cap.isOpened():
            ret, frame = cap.read()
            im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            im, boxes = extract_faces(im, "0", device)
            if im is not None and len(im) == 1:
                im[0].save(self.processedFramesPath + str(i) + '.jpg')
            else:
                print(f"[ERROR] Can't find faces on image {i}")
                error_frames.append(i)
            if not ret:
                break
            pbar.update(1)
            i += 1
        pbar.close()
        cap.release()
        cv2.destroyAllWindows()
        return error_frames


