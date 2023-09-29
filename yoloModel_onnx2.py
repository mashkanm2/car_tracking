

import cv2
import numpy as np
import onnxruntime as ort
import torch

from ultralytics.utils import yaml_load
from ultralytics.utils.checks import check_requirements

from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS

class Yolov8:

    def __init__(self, model_path, confidence_thres, iou_thres,detect_obj=None,
                yml_path='./yolov8_data.yml',
                reid_model='./weights/osnet_x0_25_msmt17.pt',
                tracking_method='botsort',
                half=False,per_class=False):
        """
        Initializes an instance of the Yolov8 class.

        Args:
            onnx_model: Path to the pt model.
            confidence_thres: Confidence threshold for filtering detections.
            iou_thres: IoU (Intersection over Union) threshold for non-maximum suppression.
            detect_obj: list of detection object for filter on output
        """
        # Check the requirements and select the appropriate backend (CPU or GPU)
        check_requirements('onnxruntime-gpu' if torch.cuda.is_available() else 'onnxruntime')
        
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.detect_obj_list=detect_obj
        # Load the class names from the COCO dataset
        self.classes = yaml_load(yml_path)['names']
        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))
        # Create an inference session using the ONNX model and specify execution providers
        self._session = ort.InferenceSession(str(model_path), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

        # Get the model inputs
        self.model_inputs = self._session.get_inputs()

        # Store the shape of the input for later use
        input_shape = self.model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]
        
        self._create_traker(tracking_method,reid_model,half,per_class)
    
    def _create_traker(self,tracking_method,reid_model,half,per_class):
        tracking_config = \
            ROOT / 'boxmot' / 'configs' /(tracking_method + '.yaml')
            
        self.tracker = create_tracker(
            tracking_method,
            tracking_config,
            reid_model,
            torch.device('cpu'),
            half,
            per_class
        )
        # motion only modeles do not have
        if hasattr(self.tracker, 'model'):
            self.tracker.model.warmup()
   
    def draw_detections(self, img, box, score, class_id,info=None):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """

        # Extract the coordinates of the bounding box
        x1, y1, x2, y2 = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        # Create the label text with class name and score
        if info:
            label = f'tID:{info}-{self.classes[class_id]}: {score:.2f}'
        else:
            label = f'{self.classes[class_id]}: {score:.2f}'

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = int(x1)
        label_y = int(y1 - 10 if y1 - 10 > label_height else y1 + 10)

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color,
                      cv2.FILLED)

        # Draw the label text on the image
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    
    def preprocess(self,img_in):
        """
        Preprocesses the input image before performing inference.
            img_in : np.ndarray input image

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)

        # Resize the image to match the input shape
        img = cv2.resize(img, (self.input_width, self.input_height))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return image_data
    
    def postprocess(self, output):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """

        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        dets=[]

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.confidence_thres:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)
                if isinstance(self.detect_obj_list,list) and (not(class_id in self.detect_obj_list)):
                    continue
                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                # Add the class ID, score, and box coordinates to the respective lists
                # dets = [[x1,y1,x2,y2,conf,class_ID]]
                dets.append([left,top,left+width,top+height,max_score,class_id])
        
        dets=np.array(dets)
        if len(dets)<1:
            return [],[]
        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(dets[:,0:4], dets[:,4], self.confidence_thres, self.iou_thres)
        dets=dets[indices,:]
        
        return indices,dets
    
    
    @torch.no_grad()
    def predict_img(self,img_in):
        # Preprocess the image data
        img_data = self.preprocess(img_in)

        # Run inference using the preprocessed image data
        outputs = self._session.run(None, {self.model_inputs[0].name: img_data})
        # Perform post-processing on the outputs to obtain output image.
        indices,dets = self.postprocess(outputs)  # output image
        
        online_targets = self.tracker.update(dets,img_in)
        # online_targets=[x1,y1,x2,y2,target_ID,class,score]
        for target in online_targets:
            # Get the box, score, and class ID corresponding to the index
            box = target[0:4]
            t_ID = int(target[4])
            t_config=target[5]
            class_id = int(target[6])
            # # Draw the detection on the input image
            self.draw_detections(img_in, box, t_config, class_id,t_ID)
    
        return img_in



if __name__ == '__main__':

    
    YOLO_ARGS = {
        "model_path":WEIGHTS / 'yolov8m_2.onnx',        # Input your ONNX model.
        "yml_path":WEIGHTS / 'yolov8_data.yml',         # Input your yml path.
        "confidence_thres":0.2,                         # Confidence threshold
        "iou_thres":0.2,                                # NMS IoU threshold
        'reid_model':WEIGHTS / 'osnet_x0_25_msmt17.pt', #reid model path
        'tracking_method':'botsort',                 # deepocsort, botsort, strongsort, ocsort, bytetrack
        'half':False,                                   #use FP16 half-precision inference
        'per_class':False,                              #not mix up classes when tracking
        'detect_obj':None                               # Filter objects
    }

    # Check the requirements and select the appropriate backend (CPU or GPU)
    check_requirements('onnxruntime-gpu' if torch.cuda.is_available() else 'onnxruntime')
    
    # Create an instance of the Yolov8 class with the specified arguments
    DETECTION = Yolov8(**YOLO_ARGS)

    
    video_path_='./datas/test9.mp4'              # Path to input video.
    save_path_video='./datas/output_004.avi'       # Path to output video.

    cap=cv2.VideoCapture(video_path_)
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        raise("Error opening video stream or file")
    
    ret,img_in=cap.read()
    if ret:
        # Get the height and width of the input image
        DETECTION.img_height, DETECTION.img_width = img_in.shape[:2]
    
    out_img_h=DETECTION.img_height//2
    out_img_w=DETECTION.img_width//2
    
    out = cv2.VideoWriter(save_path_video,cv2.VideoWriter_fourcc(*'DIVX'),30, (out_img_w,out_img_h))

    # Read until video is completed
    while(cap.isOpened()):
        ret,img_in=cap.read()
        if ret:

            # Run inference using the preprocessed image data
            out_img = DETECTION.predict_img(img_in)

            # resize image as frame size
            output_image=cv2.resize(out_img,[out_img_w,out_img_h])
            # Display the output image in a window
            cv2.imshow('Output', output_image)
            
            out.write(output_image)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            # break
        
        else:
            break
    
    cv2.destroyAllWindows()
    out.release()
    cap.release()



