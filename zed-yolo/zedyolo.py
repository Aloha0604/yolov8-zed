import cv2
import torch
import argparse
import math
import pyzed.sl as sl
from ultralytics import YOLO

def main():
    # 初始化ZED相机
    zed = sl.Camera()
    
    # 创建初始化参数对象，并设置深度模式和单位
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    
    # 打开相机，并检查是否成功
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print("无法打开ZED相机：", repr(status))
        exit()

    # 创建YOLO模型
    model = YOLO('yolov8n.pt')

    # 创建显示窗口并设置大小
    cv2.namedWindow('00', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('00', 1280, 360)
    # 创建视频输出对象
    out_video = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (2560, 720))

    # 创建运行时参数对象
    runtime_parameters = sl.RuntimeParameters()
    image = sl.Mat()  # 用于存储相机图像的对象
    depth = sl.Mat()  # 用于存储深度图的对象
    point_cloud = sl.Mat()  # 用于存储点云数据的对象
    
    i = 0
    while i < 500:  # 循环处理50帧图像，可根据需要调整
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # 获取图像和深度信息
            zed.retrieve_image(image, sl.VIEW.LEFT)  # 获取左眼图像
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)  # 获取深度图
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)  # 获取点云数据
            
            # 将ZED相机的图像数据转换为适用于YOLO检测的格式
            im0 = image.get_data()[:, :, :3]
            results = model.track(im0, save=False, conf=0.5)  # 使用YOLO模型进行目标检测
            annotated_frame = results[0].plot()  # 获取带注释的帧图像
            boxes = results[0].boxes.xywh.cpu()  # 获取检测到的框的坐标
            
            for box in boxes:
                x_center, y_center, width, height = box.tolist()  # 获取框的中心点和宽高
                x1 = x_center - width / 2  # 计算框的左上角x坐标
                y1 = y_center - height / 2  # 计算框的左上角y坐标
                x2 = x_center + width / 2  # 计算框的右下角x坐标
                y2 = y_center + height / 2  # 计算框的右下角y坐标
                
                if 0 < x2 < im0.shape[1]:  # 检查框是否在图像内
                    err, point_cloud_value = point_cloud.get_value(int(x_center), int(y_center))  # 获取中心点的3D坐标
                    if math.isfinite(point_cloud_value[2]):  # 检查深度值是否有效
                        # 计算欧几里得距离并转换为米
                        distance = math.sqrt(point_cloud_value[0]**2 + point_cloud_value[1]**2 + point_cloud_value[2]**2) / 1000
                        text_dis_avg = f"distance: {distance:.2f}m"  # 格式化距离字符串
                        # 在图像上显示距离
                        cv2.putText(annotated_frame, text_dis_avg, (int(x2 + 5), int(y1 + 30)), cv2.FONT_ITALIC, 1.2, (0, 255, 255), 3)
                        print(f"目标在 ({x_center},{y_center}) 处的距离: {distance} 米")
                    else:
                        print(f"无法计算 ({x_center},{y_center}) 处的距离")

            # 显示带注释的帧
            cv2.imshow('00', annotated_frame)
            # 将带注释的帧写入输出视频
            out_video.write(annotated_frame)

            # 检查用户是否按下 'q' 键以退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            i += 1

    # 释放资源
    out_video.release()
    cv2.destroyAllWindows()
    zed.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='模型路径')
    parser.add_argument('--svo', type=str, default=None, help='可选的svo文件')
    parser.add_argument('--img_size', type=int, default=416, help='推理图像大小（像素）')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='目标置信度阈值')
    opt = parser.parse_args()

    with torch.no_grad():
        main()
