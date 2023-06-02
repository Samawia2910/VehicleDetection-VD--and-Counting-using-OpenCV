import cv2
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import numpy as np
sample_image = cv2.imread('3d-shapes.jpg')
img = cv2.cvtColor(sample_image,cv2.COLOR_BGR2RGB)
img = cv2.resize(img,(256,256))
#applying threshold
# gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
# _,thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
#detecting edges
edges = cv2.dilate(cv2.Canny(thresh,0,255),None)
#Detecting Contours To Create Mask
cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
mask = np.zeros((256,256), np.uint8)
masked = cv2.drawContours(mask, [cnt],-1, 255, -1)
#Segmenting the Regions
dst = cv2.bitwise_and(img, img, mask=mask)
segmented = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(segmented)
cv2.imshow("Image", segmented)
cv2.waitKey()