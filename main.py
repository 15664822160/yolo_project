import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from ultralytics import YOLO
from wxauto import WeChat
import queue
import cv2


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        #窗口大小
        self.resize(1200,800)
        #标题
        self.setWindowTitle("基于Yolov8的人体摔倒检测系统")
        #窗标
        self.setWindowIcon(QIcon("R-C.png"))
        #模型
        self.model = YOLO(r'runs/detect/train2/weights/best.pt')
        # 计数器
        self.num = 0
        #标志位，控制摄像头的开始和暂停
        self.pua = False
        #双向队列，判断是否摔倒
        self.deq=queue.Queue()
        self.deq_wf=queue.Queue()
        #概率数
        self.deq_fall=0
        self.init_ui()
        # 连接信号和槽函数


    def init_ui(self):
        #总布局器V_layout
        V_layout=QVBoxLayout()
        #上布局器s_layout
        s_layout=QHBoxLayout()
        #两个文本框
        self.l1=QLabel()
        self.l2=QLabel()
        self.l1.setFixedSize(600,400)
        self.l2.setFixedSize(600,400)
        self.l1.setStyleSheet('border:1px solid #DDDDDD;')
        self.l2.setStyleSheet('border:1px solid #DDDDDD;')
        s_layout.addWidget(self.l1)
        s_layout.addWidget(self.l2)
        #下组zu
        zu=QGroupBox()
        #下组布局器x_layout
        x_layout=QHBoxLayout()
        #下左文本框l3
        self.l3=QLabel()
        self.l3.resize(1000,20)
        self.l3.setWordWrap(True)#自动换行
        self.l3.setAlignment(Qt.AlignTop)#顶格
        # 创建一个 QScrollArea 滑动窗格
        scr = QScrollArea()
        scr.setWidgetResizable(True) #?
        # 允许滚动区域内的部件调整大小
        scr.setWidget(self.l3)  # 将 QLabel 添加到滚动区域中
        #滚动布局器
        v_layout=QVBoxLayout()
        v_layout.addWidget(scr)
        x_layout.addLayout(v_layout)
        #下右布局器xv_layout
        xv_layout=QVBoxLayout()
        p1=QPushButton("🤖模型选择")
        p2=QPushButton("🎞️视频文件")
        p3=QPushButton("📹摄像头")
        p4=QPushButton("📷照片文件")
        self.p5_p=QPushButton("🛑停止")
        p1.clicked.connect(self.p1)
        p2.clicked.connect(self.p2)
        p3.clicked.connect(self.p3)
        p4.clicked.connect(self.p4)
        self.p5_p.clicked.connect(self.p5)
        xv_layout.addWidget(p1)
        xv_layout.addWidget(p2)
        xv_layout.addWidget(p3)
        xv_layout.addWidget(p4)
        xv_layout.addWidget(self.p5_p)
        x_layout.addLayout(xv_layout)
        zu.setLayout(x_layout)
        V_layout.addLayout(s_layout)
        V_layout.addWidget(zu)
        self.setLayout(V_layout)
        self.l4=QLabel(self)


    def p1(self):
        file_dg = QFileDialog()
        file_model, _ = file_dg.getOpenFileName(self, '选择图像文件', '','模型文件 (*.pt)')
        if file_model:
            self.model = YOLO(file_model)
        else:
            print('未选择模型')


    def p2(self):
        file_dialog = QFileDialog()
        video_path, _ = file_dialog.getOpenFileName(self, '选择视频文件', '', '视频文件 (*.mp4 *.avi *.mov)')
        if video_path:
            self.p2_openvideo(video_path)
        else:
            print("未选择视频文件")

    def p2_openvideo(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if self.cap.isOpened():
            # 创建定时器
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.p2_update_frame)
            # 启动定时器，每隔30毫秒更新一次帧
            self.timer.start(30)
        else:
            print("无法打开视频文件")

    def p2_update_frame(self):
        if not self.pua:
            if self.cap.isOpened():
                # 读取一帧视频
                ret, frame = self.cap.read()
                print()
                if ret:
                    #左图
                    # 将OpenCV的BGR格式转换为RGB格式
                    frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # 获取帧的高度、宽度和通道数
                    height, width, channel = frame1.shape
                    # 计算字节数
                    bytes_per_line = 3 * width
                    # 创建QImage对象
                    q_img = QImage(frame1.data, width, height, bytes_per_line, QImage.Format_RGB888)
                    # 创建QPixmap对象
                    pixmap = QPixmap.fromImage(q_img)
                    #按比例放入l1
                    scaled_pixmap = pixmap.scaled(self.l1.width(), self.l1.height(), Qt.KeepAspectRatio)
                    # 在标签中显示图像
                    self.l1.setPixmap(scaled_pixmap)
                    #右图
                    #导入模型
                    results = self.model(frame1, conf=0.6)
                    annotated_frame = results[0].plot()  # 画框
                    info_text = ""
                    res_name=""
                    for box in results[0].boxes:
                        class_id = int(box.cls[0])
                        class_name = results[0].names[class_id]
                        res_name=class_name
                        confidence = float(box.conf[0])
                        info_text += f"类别: {class_name}, 置信度: {confidence:.2f}\n"
                    if self.deq_fall >= 100:
                        self.l4.setStyleSheet("color: red;")
                        self.l4.setText(str(self.deq_fall))
                    else:
                        self.l4.setStyleSheet("color: black;")
                        self.l4.setText(str(self.deq_fall))
                    if self.deq_fall > 120:  # 如果200帧图像中有超过60%都是摔倒，则报警
                        self.WeChat()  # 发送警报
                        self.deq = queue.Queue()
                        self.deq_wf = queue.Queue()
                        self.deq_fall=0
                    #队列判断摔倒，生成视频
                    if self.deq_wf.qsize()>200:#如果队列中的图片大于200帧
                        if not self.deq.empty():
                            self.deq.get()#去除第1个图片
                            if self.deq_wf.get()==0:#如果第1个图片是摔倒
                                self.deq_fall-=1

                    self.deq.put(frame)#新来的图片
                    if res_name=="walk":#判断图片人体站着就是1
                        self.deq_wf.put(1)
                    elif res_name=="fall":#判断图片人体倒了就是0
                        self.deq_wf.put(0)
                        self.deq_fall+=1#200帧图像中摔倒次数
                    else:
                        self.deq_wf.put(2)
                    height, width, channel = annotated_frame.shape  # 获取三维
                    bytes_per_line = 3 * width
                    # print("annotated_frame.data",type(annotated_frame.data))#其他格式
                    q_img = QImage(annotated_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                    # print("q_img",type(q_img))#QImage格式
                    pixmap = QPixmap.fromImage(q_img)
                    sc_pixmap = pixmap.scaled(self.l1.width(), self.l1.height(), Qt.KeepAspectRatio)
                    # print("pixmap",type(pixmap))#QPixmap格式

                    self.l2.setPixmap(sc_pixmap)
                    self.num += 1
                    self.l3.setText(self.l3.text() + "\n" + f"第{self.num}张图像  " + info_text)
                else:
                    # 停止定时器
                    self.timer.stop()
                    # 释放视频捕获对象
                    self.cap.release()


    def p3(self):
        self.cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("无法打开摄像头")
            return
        # 创建一个定时器，用于定时更新视频帧
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.p3_updateFrame)
        self.timer.start(30)  # 每 30 毫秒更新一次帧

    def p3_updateFrame(self):
        if not self.pua:
            ret, frame = self.cap.read()
            # print("frame",type(frame))#numpy格式
            if ret:
                # 将 BGR 格式转换为 RGB 格式
                frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 获取帧的高度、宽度和通道数
                height, width, channel = frame1.shape
                # 计算每行的字节数
                bytesPerLine = 3 * width
                # 创建 QImage 对象
                qImg = QImage(frame1.data, width, height, bytesPerLine, QImage.Format_RGB888)
                # 创建 QPixmap 对象
                pixmap = QPixmap.fromImage(qImg)
                # 在标签中显示图像
                self.l1.setPixmap(pixmap)

                #右图
                # print("frame1",type(frame1))#numpy格式
                results = self.model(frame1, conf=0.6)
                annotated_frame = results[0].plot()#画框
                info_text = ""
                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    class_name = results[0].names[class_id]
                    confidence = float(box.conf[0])
                    info_text += f"类别: {class_name}, 置信度: {confidence:.2f}\n"

                height, width, channel = annotated_frame.shape#获取三维
                bytes_per_line = 3 * width
                # print("annotated_frame.data",type(annotated_frame.data))#其他格式
                q_img = QImage(annotated_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                # print("q_img",type(q_img))#QImage格式
                pixmap = QPixmap.fromImage(q_img)
                # print("pixmap",type(pixmap))#QPixmap格式


                self.l2.setPixmap(pixmap)
                self.num+=1
                self.l3.setText(self.l3.text()+"\n"+f"第{self.num}张图像  "+info_text)


    def p4(self):
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getOpenFileName(self, '选择图片', '', '图像文件 (*.png *.jpg)')
        if file_path:
            frame_pic = cv2.imread(file_path)
            frame_pic = cv2.cvtColor(frame_pic, cv2.COLOR_BGR2RGB)
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                #Qt.KeepAspectRatio缩放图像或调整控件大小时，保持其原始的宽高比
                scaled_pixmap = pixmap.scaled(self.l1.width(), self.l1.height(), Qt.KeepAspectRatio)
                self.l1.setPixmap(scaled_pixmap)
                res = self.model(frame_pic, conf=0.6)
                ident = res[0].plot()  # 画框
                info_text = ""
                for box in res[0].boxes:
                    class_id = int(box.cls[0])
                    class_name = res[0].names[class_id]
                    confidence = float(box.conf[0])
                    info_text += f"类别: {class_name}, 置信度: {confidence:.2f}\n"

                height, width, channel = ident.shape  # 获取三维
                line = 3 * width
                img = QImage(ident.data, width, height, line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(img)
                sc_pixmap = pixmap.scaled(self.l2.width(), self.l2.height(), Qt.KeepAspectRatio)
                self.l2.setPixmap(sc_pixmap)
                self.num += 1
                self.l3.setText(self.l3.text() + "\n" + f"第{self.num}张图像  " + info_text)
            else:
                print('无法加载图像')
        else:
            print('未选择图像文件')


    def p5(self):
        self.pua=not self.pua
        if self.pua==False:
            self.p5_p.setText("🛑停止")
        else:
            self.p5_p.setText("▶️继续")

    def WeChat(self):
        def images_to_mp4(image_queue, output_path, fps=20, width=640, height=480):
            # 定义视频编码器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # 创建 VideoWriter 对象
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            while not image_queue.empty():
                # 从队列中获取图片
                image = image_queue.get()
                # 调整图片大小以匹配视频尺寸
                resized_image = cv2.resize(image, (width, height))
                # 将图片写入视频文件
                out.write(resized_image)

            # 释放 VideoWriter 对象
            out.release()

        try:
            # 获取用户主目录
            home_dir = os.path.expanduser("~")
            output_video_path = os.path.join(home_dir, "Desktop", "fall.mp4")
            # 调用函数将队列中的图片转换为 MP4 视频
            images_to_mp4(self.deq, output_video_path, fps=25, width=800, height=600)
            # 初始化微信对象
            wx = WeChat()
            video_path = output_video_path
            who = '文件传输助手'
            wx.SendMsg(msg="检测到有人摔倒", who=who)
            wx.SendFiles(filepath=(video_path,), who=who)
        except Exception as e:
            print(f"在 Wechat 方法中出现异常: {e}")



if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MyWindow()
    w.show()
    app.exec()
