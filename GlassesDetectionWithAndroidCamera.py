# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 22:39:26 2022

@author: Cemal Altan
"""

import cv2
import numpy as np
import requests

url = "http://192.168.230.66:8080/shot.jpg" 
#Here you can enter the address of the broadcast you started on your own phone camera or on a server .
#Here you only need to change the parts containing numbers .

while True:
    
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    frame = cv2.resize(img, (640, 480))
    
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    
    frame_blob = cv2.dnn.blobFromImage(frame, 1/255, (416,416), swapRB=True, crop=False)

    labels = ["Glasses"]


    
    colors = ["0,255,255","0,0,255","255,0,0","255,255,0","0,255,0"]
    colors = [np.array(color.split(",")).astype("int") for color in colors]
    colors = np.array(colors)
    colors = np.tile(colors,(18,1))


    model = cv2.dnn.readNetFromDarknet("Glassesmodel.cfg","Glasses1000.weights")

    layers = model.getLayerNames()
    output_layer = [layers[layer - 1] for layer in model.getUnconnectedOutLayers()]
    
    model.setInput(frame_blob)
    
    detection_layers = model.forward(output_layer)


    
    ids_list = []
    boxes_list = []
    confidences_list = []
    
    
    for detection_layer in detection_layers:
        for object_detection in detection_layer:
            
            scores = object_detection[5:]
            predicted_id = np.argmax(scores)
            confidence = scores[predicted_id]
            
            if confidence > 0.10:
                
                label = labels[predicted_id]
                bounding_box = object_detection[0:4] * np.array([frame_width,frame_height,frame_width,frame_height])
                (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int")
                
                start_x = int(box_center_x - (box_width/2))
                start_y = int(box_center_y - (box_height/2))
                
                
                
                ids_list.append(predicted_id)
                confidences_list.append(float(confidence))
                boxes_list.append([start_x, start_y, int(box_width), int(box_height)])
                
                
                
                
    max_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)
         
    for max_id in max_ids:
        
        max_class_id = max_id
        box = boxes_list[max_class_id]
        
        start_x = box[0] 
        start_y = box[1] 
        box_width = box[2] 
        box_height = box[3] 
         
        predicted_id = ids_list[max_class_id]
        label = labels[predicted_id]
        confidence = confidences_list[max_class_id]
      
                
        end_x = start_x + box_width
        end_y = start_y + box_height
                
        box_color = colors[predicted_id]
        box_color = [int(each) for each in box_color]
                
                
        label = "{}: {:.2f}%".format(label, confidence*100)
        print("predicted object {}".format(label))
         
                
        cv2.rectangle(frame, (start_x,start_y),(end_x,end_y),box_color,2)
        cv2.putText(frame,label,(start_x,start_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
   

    cv2.imshow("Android Camera", frame)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()