from videoCutting import *
from matplotlib import pyplot as plt

frame = cuttingVideo("trial3.mov")

trial = frame[0].copy()
imgray = cv2.cvtColor(trial, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 100, 255, 0)
contours, __ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(trial, contours, -1, (0, 0, 255), 1)
plt.imshow(trial)
plt.show()