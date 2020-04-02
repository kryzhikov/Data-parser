import os

from pafy import pafy


# Class to load video from youtube via url
# load function returns raw video directory
# TODO sound we can later extract only for appropriate parts of video from it ones

class VideoLoader(object):

    def __init__(self, url="https://www.youtube.com/watch?v=KM3S51xVS8c", rawVideoDir="./dataset/video/raw/"):
        self.url = url
        self.youtubeObj = pafy.new(url)
        self.rawVideoDir = rawVideoDir
        self.ID = self.youtubeObj.videoid
        self.rawVideo = self.youtubeObj.getbest(preftype="mp4")
        streams = self.youtubeObj.streams
        for s in streams:
            print(s.resolution, s.extension)

        if not os.path.exists(rawVideoDir):
            os.makedirs(rawVideoDir)

    def load(self):
        print("Loading file...")
        videoName = self.youtubeObj.title.replace("/", "_") + ".mp4"
        if (not os.path.exists(self.rawVideoDir + self.ID + ".mp4")) and (not os.path.exists(self.rawVideoDir + self.ID)):
            self.rawVideo.download(filepath=self.rawVideoDir)
        else:
            print("Video already loaded!")
        if os.path.exists(self.rawVideoDir + videoName):
            os.rename(self.rawVideoDir + videoName, self.rawVideoDir + self.ID + ".mp4")
        print("Done!")
