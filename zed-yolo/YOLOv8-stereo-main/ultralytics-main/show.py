import cv2

# 打开视频文件
video_path = 'output.avi'
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print("Error opening video file: ", video_path)
    exit()

# 读取视频帧并显示
while True:
    ret, frame = cap.read()
    if not ret:
        print('Finished reading the video or failed to read frame')
        break
    
    cv2.imshow('Video', frame)
    
    # 按下'q'键退出循环
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# 释放视频捕获对象和关闭窗口
cap.release()
cv2.destroyAllWindows()
