import os
import cv2
import random
import numpy as np
from gui_utils.yoloModel import Yolov8


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



ARGS = {
    "model":"./weight/yolov8s.onnx",      # Input your ONNX model.
    "yml":'./weight/yolov8_data.yml',       # Input your yml path.
    "video":'./datas/test7.mp4',            # Path to input video.
    "outputv":'./datas/output_004.avi',     # Path to output video.
    "conf_thres":0.2,                       # Confidence threshold
    "iou_thres":0.2,                        # NMS IoU threshold
}

# Create an instance of the Yolov8 class with the specified arguments
DETECTION = Yolov8(onnx_model=ARGS["model"],
                confidence_thres= ARGS["conf_thres"],
                    iou_thres= ARGS["iou_thres"],
                    detect_obj=None,
                    yml_path=ARGS["yml"])


    
def run():
    cap=cv2.VideoCapture(ARGS["video"])
    # Check if camera opened successfully
    
    
    ret,img_in=cap.read()
    if ret:
        # Get the height and width of the input image
        DETECTION.img_height, DETECTION.img_width = img_in.shape[:2]
    
    out_img_h=DETECTION.img_height//2
    out_img_w=DETECTION.img_width//2
    
    out = cv2.VideoWriter(ARGS["outputv"],cv2.VideoWriter_fourcc(*'DIVX'),30, (out_img_w,out_img_h))

    # Read until video is completed
    while(cap.isOpened()):
        ret,img_in=cap.read()
        if ret:
            # Preprocess the image data
            img_in = DETECTION.predict_img(img_in)
            # resize image as frame size
            output_image=cv2.resize(img_in,[out_img_w,out_img_h])
            # Display the output image in a window
            cv2.imshow('Output', output_image)
            
            out.write(output_image)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        else:
            break
    
    cv2.destroyAllWindows()
    out.release()
    cap.release()




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
        
    def start_cap(self,capture, fps):
        self.capture = capture
        Clock.schedule_interval(self.update, 1.0 / fps)
        
    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            frame = DETECTION.predict_img(frame)
            # convert it to texture
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            # buf = buf1.tobytes()
            image_texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # display image from the texture
            self.texture = image_texture
    
    def on_touch_down(self, touch):
        colorR = random.randint(0, 255)
        colorG = random.randint(0, 255)
        colorB = random.randint(0, 255)
        self.canvas.add(Color(rgb=(colorR / 255.0, colorG / 255.0, colorB / 255.0)))
        d = 30
        self.canvas.add(Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), size=(d, d)))
        touch.ud['line'] = Line(points=(touch.x, touch.y))
        self.canvas.add(touch.ud['line'])

    def on_touch_move(self, touch):
        # touch.ud['line'].points += [touch.x, touch.y]
        pass


class Root(FloatLayout):
    loadfile = ObjectProperty(None)
    savefile = ObjectProperty(None)
    text_input = ObjectProperty(None)
    capture=None
    play_btn_text=StringProperty("play")
    my_camera=KivyCamera(source_="img.jpg")
    
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
            # self.capture = cv2.VideoCapture(self.video_path)
            tm_p="./datas/test7.mp4"
            self.capture = cv2.VideoCapture(tm_p)
            if (self.capture.isOpened()== False): 
                self.result_text=str("Error opening video stream or file")
                exit()
            ret,img_in=self.capture.read()
            if ret:
                # Get the height and width of the input image
                DETECTION.img_height, DETECTION.img_width = img_in.shape[:2]
        
            self.my_camera.start_cap(capture=self.capture, fps=30)
            self.play_btn_text=str("stop")
        else:
            self.capture.release()
            self.capture=None
            self.play_btn_text=str("play")



class Editor(App):
    pass
    


Factory.register('Root', cls=Root)
Factory.register('LoadDialog', cls=LoadDialog)

if __name__ == '__main__':
    Editor().run()