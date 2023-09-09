import argparse
import os
import cv2
import numpy as np
import onnxruntime as ort
import torch

from ultralytics.utils import yaml_load
from ultralytics.utils.checks import check_requirements, check_yaml

from deep_sort.detection import Detection as ddet
from tracking_tools import generate_detections as gdet

from deep_sort import preprocessing
from deep_sort.tracker import Tracker
from deep_sort import nn_matching
from deep_sort.kalman_filter import KalmanFilter
from deep_sort.detection import Detection as Detection_Face


class Yolov8:

    def __init__(self, onnx_model, confidence_thres, iou_thres,detect_obj=None):
        """
        Initializes an instance of the Yolov8 class.

        Args:
            onnx_model: Path to the ONNX model.
            confidence_thres: Confidence threshold for filtering detections.
            iou_thres: IoU (Intersection over Union) threshold for non-maximum suppression.
            detect_obj: list of detection object for filter on output
        """
        self.onnx_model = onnx_model
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.detect_obj_list=detect_obj
        # Load the class names from the COCO dataset
        self.classes = yaml_load(check_yaml('coco128.yaml'))['names']
        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))
        
        # Create an inference session using the ONNX model and specify execution providers
        self._session = ort.InferenceSession(self.onnx_model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

        # Get the model inputs
        self.model_inputs = self._session.get_inputs()

        # Store the shape of the input for later use
        input_shape = self.model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]
        
        
        # tracking options
        self.traker_thresholds_={
            "distance_th":110,
            "blure_thre":40.0,
            "face_per_th":0.70,
            "quality_th":0.6,
            "blure_c_th":17
        }
        max_cosine_distance = 0.5
        nn_budget = None
        nms_max_overlap = 0.3
        file_path_trk = './mars-small128.pb'
        self.encoder_tracking=gdet.create_box_encoder(file_path_trk,batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.obj_tracker = Tracker(metric)
        self.kalman_f=KalmanFilter()
        
        

    def draw_detections(self, img, box, score, class_id):
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
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        label = f'{self.classes[class_id]}: {score:.2f}'

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

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

    def postprocess(self, input_image, output):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """

        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

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
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
        out_image=self.tracking_drawing(input_image,indices,boxes,scores,class_ids)
        return out_image
        
    
    def tracking_drawing(self,frame,indices,boxes,scores,class_ids):
        detections=[]
        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]

            # # Draw the detection on the input image
            # self.draw_detections(frame, box, score, class_id)
            x1, y1, w, h = box
            
            bbox = [int(x1),int(y1),int(x1)+int(w),int(y1)+int(h)]
            bboxdd = [int(x1),int(y1),int(w),int(h)]

            features = self.encoder_tracking(frame,np.array([bboxdd]))
            # f["face_img"]=faceimg
            detections.append(Detection_Face(bboxdd, score,"car", features[0]))

        # Call the tracker
        self.obj_tracker.predict()
        self.obj_tracker.update(detections)
        
        for track in self.obj_tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            #boxes.append([track[0], track[1], track[2], track[3]])
            # indexIDs.append(int(track.track_id))
            # counter.append(int(track.track_id))
            boxe_ = track.to_tlbr()
            print(boxe_)
            cv2.rectangle(frame, (int(boxe_[0]), int(boxe_[1])), (int(
                boxe_[2]), int(boxe_[3])), (255, 0, 0), 2)

            # class_name="face"
            # cv2.rectangle(frame, (int(boxe_[0]), int(boxe_[1]-30)), (int(boxe_[0])+(len(class_name)+len(str(track.track_id)))*17, int(boxe_[1])), color, -1)
            cv2.putText(frame, "car-" + str(track.track_id),(int(boxe_[0]), int(boxe_[1])),0, 0.75, (255,255,255),2)


        # Return the modified input image
        return frame

    def run(self,video_path,save_path=None):
        """
        Performs inference using an ONNX model and returns the output image with drawn detections.
        Args:
            video_path: path of input video
            save_path : path to save output video
        
        Returns:
            output_img: The output image with drawn detections.
        """
        
        cap=cv2.VideoCapture(video_path)
        # Check if camera opened successfully
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")
            return
        
        out = cv2.VideoWriter(save_path,cv2.VideoWriter_fourcc(*'DIVX'),30, (640,480))

        # Read until video is completed
        while(cap.isOpened()):
            ret,img_in=cap.read()
            if ret:
                # Get the height and width of the input image
                self.img_height, self.img_width = img_in.shape[:2]
                
                # Preprocess the image data
                img_data = self.preprocess(img_in)

                # Run inference using the preprocessed image data
                outputs = self._session.run(None, {self.model_inputs[0].name: img_data})
                # Perform post-processing on the outputs to obtain output image.
                output_image = self.postprocess(img_in, outputs)  # output image
                # resize image as frame size
                output_image=cv2.resize(output_image,[640,480])
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


if __name__ == '__main__':
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov8s.onnx', help='Input your ONNX model.')
    parser.add_argument('--video', type=str, default=str('./zHUt2qWqmztC0qoGGa0G_720p.mp4'), help='Path to input video.')
    parser.add_argument('--outputv', type=str, default=str('./output.avi'), help='Path to output video.')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    args = parser.parse_args()

    # Check the requirements and select the appropriate backend (CPU or GPU)
    check_requirements('onnxruntime-gpu' if torch.cuda.is_available() else 'onnxruntime')

    # Create an instance of the Yolov8 class with the specified arguments
    detection = Yolov8(args.model,args.conf_thres, args.iou_thres,detect_obj=[2,7])

    # Perform object detection and obtain the output image
    detection.run(args.video,args.outputv)
