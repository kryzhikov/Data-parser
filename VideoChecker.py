import shutil

import cv2
from tqdm import tqdm

from ffe import *
import skvideo.io
from cv2 import VideoWriter, VideoWriter_fourcc


# __init__ get a path to directory to examine

# TODO here we can also save detected faces
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


class VideoChecker(object):
    def __init__(self, directory):
        self.directory = directory

    def check(self, fa, device=None, model=None, debug=False, KP_d=None):
        '''
            Предполагаем, что на первом кадре всегда искомый спикер
            Берём его лицо как таргет
            Просматриваем все последующие кадры и ищем на них таргет спикера
            При этом в cur_v записывается найденное лицо в виде класса FFE.Face или None, если нет лица
            ближе чем ∆
        '''
        device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu') if device is None else device
        for file in (os.listdir(self.directory)):

            print('=' * 20, f"processing {file}", "=" * 20)
            if ".DS_Store" in file or ".mp4" not in file:
                continue
#             f_dir = self.directory + "/" + file[:-4] + "/"
            if os.path.exists(self.directory + "/" + file[:-4]+"_2DFull.npy"):
                print("file already checked!")
                continue
#             if not os.path.exists(self.directory + "/" + file[:-4] + "/"):
#                 os.mkdir(self.directory + "/" + file[:-4] + "/")

            correctFile = True
            cap = cv2.VideoCapture(self.directory + "/" + file)
            ret, frame = cap.read()
            if not ret:
                print("BROKEN VIDEO not ret")
                os.remove(self.directory + "/" + file)
                continue
            try:
                im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                im_p = ParsableImage(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), device=device, name=file, face_al=fa,
                                     KP_d=KP_d)
                im_p.parseFaces(margin=60, model=model)
            except Exception as ex:
                print("BROKEN VIDEO cant read imge ", ex)
                os.remove(self.directory + "/" + file)
                continue

            if im_p.faces is None or len(im_p.faces) == 0 or len(im_p.faces) > 1:
                print(f"[ERROR] Can't find faces on image !")
                os.remove(self.directory + "/" + file)
                continue
            else:
                prev_v = im_p.faces[0].getVector()

            #             im_p.faces[0].faceImage.save(self.directory + "/" + file[:-4] + "/" + str(0) + ".jpg")
            im_p.showBoxes()
            im_p.parsedImage.save(f"./{file}_Debug.jpg")
            frames =[ cv2.resize(im_p.faces[0].imgCv2, (256, 256))[:, :, ::-1]]
            dres2 = []
            dres3 = []
            BB = []
            #             np.save(self.directory + "/" + file[:-4] + "/2D" + str(0), im_p.faces[0].lms2d)
            #             np.save(self.directory + "/" + file[:-4] + "/3D" + str(0), im_p.faces[0].lms3d)
            dres2.append(im_p.faces[0].lms2d)
            dres3.append(im_p.faces[0].lms3d)
            #             torch.save(im_p.faces[0].kp_source, self.directory + "/" + file[:-4] + "/KP_D" + str(0))
#             KP_D.append(im_p.faces[0].kp_source)
            BB.append(im_p.faces[0].box_m)
            idx = 1
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            pbar = tqdm(total=length + 1)
            check_interval = 40 
            prevbb = im_p.faces[0].box_m
            
            ''' diff '''
            print('\n##### Start pre-checking #####')
            prechecker_freq = 25
            is_correct_video = True
            video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for frame_idx in range(0, video_length, prechecker_freq):
                cap.set(1, frame_idx)
                ret, frame = cap.read()
                
                # check video
                if not ret:
                    print("BROKEN VIDEO not ret")
                    break
                
                # check what? (???)
                try:
                    im_p = ParsableImage(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), device, file, face_al=fa, KP_d=KP_d)
                    im_p.parseFaces(margin=60, model=model)       
                except Exception as ex:
                    print(f"[ERROR] Can't find faces on image  Cant Parse cause of {ex}")
                    im_p.PILImage.save(f"{file}WHAT?.jpg")
                    correctFile = False
                    break
                    
                # check face)
                if len(im_p.faces) == 0:
                    print(f"[ERROR] Can't find faces on image ! No faces")
                    correctFile = False
                    break
         
                # check speaker
                cur_v = find_face_on_image(im_p.faces, prev_v, im_p.PILImage, file)
                if cur_v is None:
                    print(f"[ERROR] Can't find speaker face on image ! Speaker not found")
                    correctFile = False
                    break
                del cur_v
            print('##### Finish pre-checking #####')
            # set start frame to zero
            cap.set(1, 0)
            ''' end diff '''
                
                
            if correctFile: # diff
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        print("BROKEN VIDEO not ret")
                        break

                    try:
                        im_p = ParsableImage(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), device, file, face_al=fa, KP_d=KP_d)
                        if idx % check_interval == 0:
                            im_p.parseFaces(margin=60, model=model)
                        else:
                            im_p.parseFaces(margin=60, model=None)

                    except Exception as ex:
                        print(f"[ERROR] Can't find faces on image  Cant Parse cause of {ex}")
                        im_p.PILImage.save(f"{file}WHAT?.jpg")
                        correctFile = False
                        break

                    if len(im_p.faces) == 0:
                        print(f"[ERROR] Can't find faces on image ! No faces")
                        correctFile = False
                        break
                    if idx % check_interval == 0:


                        cur_v = find_face_on_image(im_p.faces, prev_v, im_p.PILImage, file)

                        if cur_v is None:
                            print(f"[ERROR] Can't find speaker face on image ! Speaker not found")
                            correctFile = False
                            break
                    else:
                        cur_v = im_p.faces[0]


                    curbb = cur_v.box_m
                    if bb_intersection_over_union(curbb, prevbb)<= 0: 
                        print(f"[ERROR] Can't find speaker face on image ! Lost speaker!")
                        correctFile = False
                        break
                    prevbb = curbb

                    frames.append(cv2.resize(cur_v.imgCv2, (256, 256))[:, :, ::-1])
    #                 im_p.showBoxes()
                    # im_p.parsedImage.save(f"./{file}_sample.jpg")
                    #                 np.save(self.directory + "/" + file[:-4] + "/2D" + str(idx), cur_v.lms2d)
                    #                 np.save(self.directory + "/" + file[:-4] + "/3D" + str(idx), cur_v.lms3d)
                    dres2.append(cur_v.lms2d)
                    dres3.append(cur_v.lms3d)
                    BB.append(cur_v.box_m)
                    #                 torch.save(cur_v.kp_source, self.directory + "/" + file[:-4] + "/KP_D" + str(idx))
    #                 KP_D.append(cur_v.kp_source)

                    if debug:
                        tmp = cur_v.faceImage.copy()
                        imageD = ImageDraw.Draw(tmp)
                        for (x, y) in cur_v.dlibs_m:
                            # print(x, y)
                            imageD.point((x, y), 'green')
                        tmp.save("DEBUG" + file.rstrip(".mp4") + ".jpg")

                    if idx % check_interval == 0:
                        im_p.flush()
                    idx += 1
                    del cur_v
                    # dump_tensors()
                    pbar.update(1)
            pbar.close()
#             torch.save(KP_D, self.directory + "/" + file[:-4] + "/KP_D.pt")
#             print(np.array(frames).shape)
#             fourcc = VideoWriter_fourcc(*'MP42')
#             video = VideoWriter(self.directory + "/" + file[:-4] + "/speaker.avi", fourcc, float(25), (256, 256))
#             for i in frames:
#                   video.write(i[:, :, ::-1])
#             video.release()
            #skvideo.io.vwrite(self.directory + "/" + file[:-4] + "/video.mp4", np.array(frames))
            if not correctFile:
                os.remove(self.directory + "/" + file)
                #os.remove(self.directory + "/" + file[:-4]+".txt")

            else:
                np.save(self.directory + "/" + file[:-4] + "_2DFull.npy", np.array(dres2))
                np.save(self.directory + "/" + file[:-4] + "_3DFull.npy", np.array(dres3))
        #             np.save(self.directory + "/" + file[:-4] + "/frames.npy", np.array(frames))
                np.save(self.directory + "/" + file[:-4] + "_BB.npy", np.array(BB))
        #             shutil.copyfile(self.directory + "/" + file[:-4]+".txt", self.directory + "/" + file[:-4]+"/subs.txt")
        #             shutil.copyfile(self.directory + "/" + file, self.directory + "/" + file[:-4]+"/speaket.mp4")
            print("CORRECT?", correctFile)
            print("FNAME", file)
            cap.release()
            del prev_v
            
