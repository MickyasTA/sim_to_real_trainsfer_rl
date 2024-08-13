#!/usr/bin/env python3

""" This script sets up a ROS node that captures images from a camera, sends 
    them to a server for processing, and then receives wheel commands from the
    server to control the robot's wheels. It includes handling for ROS node 
    shutdown to ensure the robot stops moving when the node is terminated.
    
    In the callback method, log the time the image was published (image_publish_time)
    and the time it was received by the client (current_time).

    In the publish_wheels method, log the time the action command was received 
    (action_receive_time) and the time it was executed (action_execute_time).

    """
import time
import rospy
import cv2
import requests # Library for making HTTP requests.
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import WheelsCmdStamped
from cv_bridge import CvBridge
import os

# URL of the server that processes the images and returns wheel commands.
SERVER_URL = "http://192.168.2.25:5000/process_image" # "http://192.168.110.25:5000/process_image"

class CameraClientNode:
    def __init__(self):
        self.bridge = CvBridge() # Create a CvBridge object to convert ROS messages(images) to OpenCV images.
        self.vehicle_name = os.environ['VEHICLE_NAME']

        # Defines the topic for receiving compressed images from the camera.
        self.camera_topic = f"/{self.vehicle_name}/camera_node/image/compressed"
        # Defines the topic for publishing the wheel commands to the robot.
        self.wheels_topic = f"/{self.vehicle_name}/wheels_driver_node/wheels_cmd"
        
        self.pub = rospy.Publisher(self.wheels_topic, WheelsCmdStamped, queue_size=1)# Create a publisher for the wheel commands.
        self.sub = rospy.Subscriber(self.camera_topic, CompressedImage, self.callback)# Create a subscriber for the camera images.
        
        self.latent_vectors = [] # Placeholder for accumulating latent vectors.
        self.frame_skip_interval = rospy.Duration(0.1) # Interval for accumulating frames.
        self.prev_image_time = rospy.Time.now() # Time of the previous frame.

        self.image_publish_time = None  # Initialize image publish time

        rospy.on_shutdown(self.on_shutdown) # Register the shutdown callback.

    def callback(self, msg):
        
        # Image published time
        image_publish_time = msg.header.stamp.to_sec() # Time for the current frame.
        image_received_time = time.time() # Current time.
        rospy.loginfo(f"Image published at: {image_publish_time}, Image received at: {image_received_time}")

        image = self.bridge.compressed_imgmsg_to_cv2(msg) # Convert the ROS compressed image message to an OpenCV image.
        _, img_encoded = cv2.imencode('.jpg', image) # Encode the image in JPEG format.
        files = {'file': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')} # Prepares the encoded image for HTTP POST request.
        #response = requests.post(SERVER_URL, files=files) # Sends the image to the server for processing.
        
        try:
            server_response_start_time = time.time() # Time for sending the image to the server.
            response = requests.post(SERVER_URL, files=files) # Sends the image to the server for processing. 
            server_response_end_time = time.time() # Time for receiving the response from the server.
            rospy.loginfo(f"Server response time: {server_response_end_time - server_response_start_time} seconds")


            if response.status_code == 200: # If the request is successful, extract the wheel commands from the response.
                data = response.json() # Extracts the JSON data from the response.
                action_received_time = time.time()
                rospy.loginfo(f"Action received at: {action_received_time}")
                self.publish_wheels(data['vel_left'], data['vel_right']) # Publishes the wheel commands to the robot.
            else:
                rospy.logerr(f"Failed to process image: {response.status_code}") # Logs an error if the request fails.
        except requests.exceptions.RequestException as e:
            rospy.logerr(f"Request error: {e}") # Logs an error if there is an exception during the request.

    def publish_wheels(self, vel_left, vel_right): # Publishes the wheel commands received from the server. 

        action_execute_time = time.time() # Time for receiving the wheel commands.
        rospy.loginfo(f"Action received at: {action_execute_time}") # Logs the time of receiving the wheel commands.

        msg = WheelsCmdStamped() # Creates a new WheelsCmdStamped message.
        msg.vel_left = vel_left # Sets the left wheel velocity.
        msg.vel_right = vel_right # Sets the right wheel velocity.
        self.pub.publish(msg)

        # Calculate total delay
        if self.image_publish_time is not None:
            total_delay = action_execute_time - self.image_publish_time
            rospy.loginfo(f" ************* Total delay from image publish to action execute: {total_delay} seconds *************")

    def on_shutdown(self):
        rospy.loginfo("Shutting down, stopping the robot.")
        self.publish_wheels(0, 0)  # Stop the wheels

if __name__ == '__main__':
    rospy.init_node('camera_client_node', anonymous=True)
    node = CameraClientNode()
    rospy.spin()
