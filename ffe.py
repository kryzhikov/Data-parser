from __future__ import division

import collections
from skimage.transform import resize

import face_alignment
from facenet_pytorch import fixed_image_standardization
from torchvision import transforms

from utils import *

plot_style = dict(marker='o',
                  markersize=4,
                  linestyle='-',
                  lw=2)

pred_type = collections.namedtuple('prediction_type', ['slice', 'color', 'color_str'])
pred_types = {'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5), 'blue'),
              'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4), 'red'),
              'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4), 'red'),
              'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4), 'red'),
              'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4), 'red'),
              'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3), 'yellow'),
              'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3), 'yellow'),
              'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3), 'green'),
              'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4), 'green')
              }


class Face(object):
    def __init__(self, faceImage, box, box_m, sourceImage, fa = None, label=None, device = None, transforms_ = None, KP_d = None):
        self.faceImage = faceImage
        self.box = box
        self.box_m = box_m
        self.sourceImage = sourceImage
        self.imgCv2 = np.asarray(self.faceImage)[:, :, ::-1].copy()
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.device_str = 'cpu' if self.device == torch.device('cpu') else 'cuda:0'

        self.fa = fa if fa is not None else face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device = self.device_str)

        lms = self.fa.get_landmarks(self.imgCv2[:, :, ::-1])[-1]
        self.lms2d = np.array([[p[0], p[1]] for p in lms])
        self.lms3d = lms
        self.model = None
        self.transforms_ = transforms.Compose([
            transforms.Resize((256, 256)),
            np.float32,
            transforms.ToTensor(),
            fixed_image_standardization
        ]) if transforms_ is None else transforms_
        self.faceTensor = self.transforms_(faceImage)
        self.label = label
        self.globallms3d = np.array([[p[0] + self.box_m[0], p[1] + self.box_m[1], p[2]] for p in lms])

        self.globallms2d = np.array([[p[0] + self.box_m[0], p[1] + self.box_m[1]]for p in lms])
        s = resize(self.imgCv2, (256, 256))[..., :3]
        s = torch.tensor(s[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        self.kp_source = KP_d(s)

    def showImg(self):
        imshow(image=self.faceTensor, title=self.label)

    def setModel(self, model):
        if self.model is None:
            self.model = model.to(self.device)
        else:
            if input("Sure want to change current model?[y/n]") == "y":
                self.model = model.to(self.device)

    def buildVec(self, debug=False, tensor=False):
        face = [self.transforms_(self.faceImage)]
        if debug:
            self.showImg()
        if self.model is None:
            print("No model set. Run set_model()")
            return None
        self.batch = get_batch(face, self.device)
        vec = self.model(self.batch)[0]
        self.vector = vec

    def getVector(self, tensor=False):
        if self.vector is None:
            print("No vector built, run Face.build_vec() to build vector for face")
        else:
            return self.vector.cpu().detach().numpy() if not tensor else self.vector

    def showLandmarks(self):
        preds = self.lms3d
        fig = plt.figure(figsize=plt.figaspect(.5))
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(self.imgCv2[:, :, ::-1])
        for pred_type in pred_types.values():
            ax.plot(preds[pred_type.slice, 0],
                    preds[pred_type.slice, 1],
                    color=pred_type.color, **plot_style)

        ax.axis('off')
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        surf = ax.scatter(preds[:, 0] * 1.2,
                          preds[:, 1],
                          preds[:, 2],
                          c='cyan',
                          alpha=1.0,
                          edgecolor='b')

        for pred_type in pred_types.values():
            ax.plot3D(preds[pred_type.slice, 0] * 1.2,
                      preds[pred_type.slice, 1],
                      preds[pred_type.slice, 2], color='blue')

        ax.view_init(elev=90., azim=90.)
        ax.set_xlim(ax.get_xlim()[::-1])
        plt.show()

    def flush(self):
        del self.model
        torch.cuda.empty_cache()
        del self.vector
        torch.cuda.empty_cache()
        for i in self.batch:
            del i
            torch.cuda.empty_cache()


class ParsableImage():
    def __init__(self, sourceImage, device: object = None, name: str = "best_photo", face_al = None, KP_d=None):
        self.sourceImage= sourceImage
        self.PILImage = Image.fromarray(self.sourceImage)
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.device_str = 'cpu' if self.device == torch.device('cpu') else 'cuda:0'
        self.fa = face_al if face_al is not None else face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device = self.device_str)
        self.name = name
        self.faceBoxes = []
        self.faces = []
        self.parsedImage = self.PILImage.copy()
        self.KP_d = KP_d

    def parseFaces(self, margin, model = None) -> object:
        faceImages, self.faceBoxes, self.faceBoxes_margined = extract_faces(self.sourceImage, self.fa, margin = margin)
        # lms = self.fa.get_landmarks(self.sourceImagepath, self.faceBoxes)
        if faceImages is None or len(faceImages) == 0 :
            return None
        for i, b, b_m in (zip(faceImages, self.faceBoxes, self.faceBoxes_margined)):
            tmpFace = Face(i, b, b_m, self.PILImage, self.fa, device=self.device, KP_d = self.KP_d)
            if model is not None:
                tmpFace.setModel(model)
                tmpFace.buildVec(tensor = True)
            self.faces.append(tmpFace)

    def showBoxes(self):
        img_draw = ImageDraw.Draw(self.parsedImage)
        for i in self.faces:
            img_draw.rectangle(i.box_m, width=4,
                               outline='black')
            preds = i.globallms3d
            for pred_type in pred_types.values():
                for i in preds[pred_type.slice]:
                    img_draw.ellipse([i[0], i[1], i[0]+self.PILImage.size[0]/500,i[1]+self.PILImage.size[1]/500], fill=pred_type.color_str)

        self.parsedImage.show()


    def flush(self):
        for i in self.faces:
            i.flush()
