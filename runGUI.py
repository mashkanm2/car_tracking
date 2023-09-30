import os
import cv2
import random
import numpy as np
# from yoloModel_onnx import Yolov8
from yoloModel_pt import Yolov8


from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.factory import Factory
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.properties import StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Color, Ellipse, Line
import time

from boxmot.utils import ROOT, WEIGHTS


YOLO_ARGS = {
    "model_path":WEIGHTS / 'best.pt',               # Input your torch model.
    "yml_path":WEIGHTS / 'yolov8_data.yml',         # Input your yml path.
    "confidence_thres":0.2,                         # Confidence threshold
    "iou_thres":0.2,                                # NMS IoU threshold
    'reid_model':WEIGHTS / 'osnet_x0_25_msmt17.pt', #reid model path
    'tracking_method':'botsort2',                 # deepocsort, botsort, strongsort, ocsort, bytetrack
    'half':False,                                   #use FP16 half-precision inference
    'per_class':False,                              #not mix up classes when tracking
    'detect_obj':None                               # Filter objects
}

# Create an instance of the Yolov8 class with the specified arguments
DETECTION = Yolov8(**YOLO_ARGS)


class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)

class KivyCamera(Image):
    def __init__(self,source_,**kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        
        img = cv2.imread(source_, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        buf1 = cv2.flip(img, 0)
        buf = buf1.tostring()
        # buf = buf1.tobytes()
        image_texture = Texture.create(
            size=(img.shape[1], img.shape[0]), colorfmt='bgr')
        image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # display image from the texture
        self.texture = image_texture
        
        
    def start_cap(self,capture, fps,video_writer):
        self.capture = capture
        # self.video_writer=video_writer
        Clock.schedule_interval(self.update, 1.0 / fps)
        
    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            frame2 = DETECTION.predict_img(frame)
            # self.video_writer.write(frame2)
            # time.sleep(0.03)
            # convert it to texture
            buf1 = cv2.flip(frame2, 0)
            buf = buf1.tostring()
            # buf = buf1.tobytes()
            image_texture = Texture.create(
                size=(frame2.shape[1], frame2.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # display image from the texture
            self.texture = image_texture
    
    def on_touch_down(self, touch):
        # colorR = random.randint(0, 255)
        # colorG = random.randint(0, 255)
        # colorB = random.randint(0, 255)
        # self.canvas.add(Color(rgb=(colorR / 255.0, colorG / 255.0, colorB / 255.0)))
        # d = 30
        # self.canvas.add(Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), size=(d, d)))
        # touch.ud['line'] = Line(points=(touch.x, touch.y))
        # self.canvas.add(touch.ud['line'])
        y_loc,x_loc=touch.y,touch.x
        src_y,src_x=DETECTION.img_height, DETECTION.img_width
        disp_y,disp_x=600.0,800.0
        if (src_x/disp_x)>=(src_y/disp_y):
            k_d=src_x/disp_x
            
            h_001=int(src_y/k_d)
            padding_h=int((abs(h_001-disp_y)/2))
            y_loc2,x_loc2=disp_y-y_loc,disp_x-x_loc
            y_loc2=y_loc2-padding_h
            
            x_img=int(x_loc2*k_d)
            y_img=int(y_loc2*k_d)
        else:
            k_d=src_y/disp_y
            
            w_001=int(src_x/k_d)
            padding_w=int((abs(w_001-disp_x)/2))
            y_loc2,x_loc2=disp_y-y_loc,disp_x-x_loc
            x_loc2=x_loc2-padding_w
            
            x_img=int(x_loc2*k_d)
            y_img=int(y_loc2*k_d)
            
            
        DETECTION.choose_object(x_img,y_img)
        print(f"x: {touch.x}  y: {touch.y}  x_img: {x_img}  y_img: {y_img} ")

    def on_touch_move(self, touch):
        # touch.ud['line'].points += [touch.x, touch.y]
        pass


class Root(FloatLayout):
    loadfile = ObjectProperty(None)
    savefile = ObjectProperty(None)
    text_input = ObjectProperty(None)
    capture=None
    play_btn_text=StringProperty("play")
    my_camera=KivyCamera(source_="datas/img.jpg")
    
    result_text=StringProperty()

    video_path=""
    
    def __init__(self,*args,**kwargs):
        super(Root, self).__init__(*args,**kwargs)

        img_layout = BoxLayout(width=640)
        img_layout.add_widget(self.my_camera)
        
        self.add_widget(img_layout,)
    
    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()


    def load(self, path, filename):
        self.video_path=os.path.join(path, filename[0])
        self.result_text=str(self.video_path)
        self.dismiss_popup()
    
    
    def play_video(self):
        if self.capture is None:
            
            self.video_path="./datas/test7.mp4"
            
            self.capture = cv2.VideoCapture(self.video_path)
            if (self.capture.isOpened()== False): 
                self.result_text=str("Error opening video stream or file")
                exit()
            ret,img_in=self.capture.read()
            if ret:
                # Get the height and width of the input image
                DETECTION.img_height, DETECTION.img_width = img_in.shape[:2]
            
            # write_video_path=self.video_path[:-4]+"_output_.avi"
            # print("dfdfdfdffdfdfdfdfdf --------",write_video_path)
            # self.video_writer = cv2.VideoWriter(write_video_path,cv2.VideoWriter_fourcc(*'DIVX'),30, img_in.shape[:2])
            self.video_writer=None
            
            self.my_camera.start_cap(capture=self.capture, fps=30,video_writer=self.video_writer)
            self.play_btn_text=str("stop")
        else:
            # self.video_writer.release()
            self.capture.release()
            self.capture=None
            self.play_btn_text=str("play")



class Editor(App):
    pass
    


Factory.register('Root', cls=Root)
Factory.register('LoadDialog', cls=LoadDialog)

if __name__ == '__main__':
    Editor().run()