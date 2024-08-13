#!/usr/bin/env python3

import os
import rospy
import cv2
import torch
import torch.nn as nn
import numpy as np
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Encoder Model Definition
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        #flat_size = 32 * 4 * 4

        self.features = nn.Sequential(nn.Conv2d(1, 32, 2, stride=2),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(32),
                                     nn.Conv2d(32, 32, 2, stride=2),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(32),
                                     nn.Conv2d(32, 32, 2, stride=2),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(32),
                                     nn.Conv2d(32, 32, 2, stride=2),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(32))
                                                                     
        #self._out_features = (32, 3, 5)       

    def forward(self, x):
        return self.features(x)            
 
    def load_weights(self, weight_path):
        self.load_state_dict(torch.load(weight_path))


# Use Cases: Application-Specific Logic
class EncoderModel:
    def __init__(self, model_path):
        self.model = Encoder()
        self.model.eval()
        self.model.load_weights(model_path)
        self.model.to(device)

    def encode(self, image):
        if len(image.shape) == 2:  # (H, W)
            image = image[np.newaxis, np.newaxis, :, :]
        elif len(image.shape) == 3:  # (H, W, C)
            image = image.transpose(2, 0, 1)
            image = image[np.newaxis, :]
        elif len(image.shape) == 4:  # (B, C, H, W)
            pass
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")

        image_tensor = torch.from_numpy(image).float().to(device)
        with torch.no_grad():
            latent_vector = self.model(image_tensor)
        return latent_vector.squeeze().cpu().numpy()

class CameraReaderNode(DTROS):
    def __init__(self, node_name):

        # initialize the DTROS parent class
        super(CameraReaderNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        
        # static parameters
        self._vehicle_name = os.environ.get('VEHICLE_NAME', 'marsbot')
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._latent_topic = f"/{self._vehicle_name}/latent_vector"
        
        # bridge between OpenCV and ROS
        self._bridge = CvBridge()
        
        # load the encoder model
        self._encoder = EncoderModel('/code/catkin_ws/src/my-ros-project/weights/Encoder/encoder_weights1.pth')
        
        # construct subscriber and publisher
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)
        self.pub = rospy.Publisher(self._latent_topic, Float32MultiArray, queue_size=1)
        
        # list to accumulate latent vectors
        self.latent_vectors = []
        
        self.prev_image_time = rospy.Time.now() # time of the last image processed 
        self.frame_skip_interval = rospy.Duration(0.1)  # 0.1 seconds interval of skipping frames 

    def callback(self, msg): 

        # check if the time interval has passed since the last image was processed
        current_time = rospy.Time.now()
        if current_time - self.prev_image_time < self.frame_skip_interval:
            return
        self.prev_image_time = current_time # update the time of the last image processed 

        # convert JPEG bytes to CV image
        image = self._bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='mono8') # 'mono8' is the encoding for grayscale images 
        
        # encode the image to latent space vector
        image = cv2.resize(image, (80, 60))
        
        print("camera*********************",image.shape)
        
        latent_vector = self._encoder.encode(image) # latent vector is a 1D numpy array of size 2048
        
        # accumulate latent vectors
        self.latent_vectors.append(latent_vector)

        # check if we have accumulated 4 vectors before publishing them to the topic 

        if len(self.latent_vectors) == 4:

            # Concatenate the 4 latent vectors along the channel dimension
            concatenated_tensor = np.concatenate(self.latent_vectors, axis=1)  # Shape will be (1, 128, H, W)

            # Flatten and convert to Float32MultiArray
            flattened_data = concatenated_tensor.flatten()
            latent_msg = Float32MultiArray(data=flattened_data)

            self.pub.publish(latent_msg)  # publish the latent vector message to the topic
            
            # clear the list for the next batch
            self.latent_vectors = []


if __name__ == '__main__':

    # create the node
    node = CameraReaderNode(node_name='camera_reader_node') # create a node with the name 'camera_reader_node' CameraReaderNode

    # keep spinning
    rospy.spin()
