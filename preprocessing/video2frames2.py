import cv2
vidcap = cv2.VideoCapture('videos/s1/bbaf2n.mpg')
success,image = vidcap.read()
count = 0
success = True
while success:
  cv2.imwrite("frames/s1/bbaf2n%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1