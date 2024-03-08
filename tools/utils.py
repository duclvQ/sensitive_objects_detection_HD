import numpy as np 
import cv2
from PIL import Image
import imagehash

###########-HASHING-##############
def hashing_image(imge_path):
	hashed_img = imagehash.average_hash(Image.open(imge_path)) 
	return hashed_img

def are_2images_similar(hash0, hash1, cutoff):
	if hash0 - hash1 < cutoff:
  		return True
	else:
  		return False







