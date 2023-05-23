import onnxruntime
import numpy as np
import cv2 
import os 

class_dir = { 0: 'headphone', 1 : 'laptop', 2: 'phone'}

class ecommerse:

    def __init__(self,file_path):
        self.file_path = file_path 

    def prediction(self,):
        # load the model 
        model = os.path.join('model','model.onnx')
        session = onnxruntime.InferenceSession(model)
        input_name = session.get_inputs()[0].name

        #read the image
        image = cv2.imread(self.file_path)
        #resize it 
        image = cv2.resize(image,(150,150)) #!50,150 coz our model was trained on this size input 
        #convert it into numpy array 
        image = np.array(image)
        #Noramlize it and change to float 
        image = image.astype(np.float32)/225.0
        #Expand the dimension to add batch size 
        image = np.expand_dims(image,axis=0)
        #make prdiction 
        output = session.run(None, {input_name: image})
        #get the class 
        class_ = np.argmax(output[0][0])
        return class_dir[class_]


