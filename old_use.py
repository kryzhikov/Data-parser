import face_alignment
import imageio
import torch
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1

from ffe import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval()
device_str = 'cpu' if device == torch.device('cpu') else 'cuda:0'
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device=device_str)

i = imageio.imread("319360fc-13b9-4ee4-a3c9-ee25f4e12353.jpeg")
print(i)
iP = ParsableImage(i, device, "319360fc-13b9-4ee4-a3c9-ee25f4e12353.jpeg", fa)
iP.parseFaces(margin = 10)
iP.showBoxes()
