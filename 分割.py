import cv2

im0 = cv2.imread("C:/Users/bigfatcat/Desktop/a6.jpg")

cv2.imshow('test',im0)

height_0, width_0 = im0.shape[0:2]
print(im0.shape)

iml = im0[0:int(height_0), 0:int(width_0/2)]
imr = im0[0:int(height_0), int(width_0/2):int(width_0) ]
cv2.imshow('test1',iml)
cv2.imshow('test2',imr)
cv2.imwrite('iml.jpg',iml)
cv2.imwrite('imr.jpg',imr)

print(im0.shape,iml.shape,imr.shape)

cv2.waitKey(0)
cv2.destroyAllwindows()