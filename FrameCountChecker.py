import os
import shutil


class FrameCountChecker(object):
    def __init__(self, rawVideoDir="./dataset/video/raw/"):
        self.rawVideoDir = rawVideoDir

    def check(self, number):
        print("Do you sure? This operation can remove part of your data!")
        print("y/n")
        ans = input()
        if ans == "y":
            print("Checking...")
            raw_dir = os.listdir(self.rawVideoDir)
            for elem in raw_dir:
                path = self.rawVideoDir + elem
                if ".mp4" not in path:#path.count(".mp4") == 0:
                    dir_level_1_path = path
                    dir_level_1 = os.listdir(path)
                    for element in dir_level_1:
                        path = dir_level_1_path + "/" + element
                        if ".mp4" not in path:#.count(".mp4") == 0:
                            dir = os.listdir(path)
                            frames_num = 0
                            for name in dir:
                                if ".jpg" in name:#.count(".jpg") == 1:
                                    frames_num +=1
                            if frames_num != number:
                                shutil.rmtree(path)
                                if os.path.exists(path + ".mp4"):
                                    os.remove(path + ".mp4")
        print("Done!")
