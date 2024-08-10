
    # ############### Docker Images with Duckietown #################### #
    
*   usally after saving some changes in the docker project we can go to 
    that directory on our terminal and buld the project so that we can 
    create an image of that project by running this command. < dts devel build -f >

*   Now to run the project we use < dts devel run >

*   If we want to build the image on the robot we use < dts devel build -f -H marsbot >

*   If we want to run the image on the robot we use < dts devel run -H marsbot >


# ######################## The ROS REAL #############################
# Go to the my-ros-project direcory and rin the following in 2 different terminals 

chmod +x ./packages/my-package/src/CameraReaderNode.py
chmod +x ./packages/my-package/src/WheelControlNode.py

dts devel build -H marsbot -f

dts devel run -H marsbot -L WheelControlNode
dts devel run -H marsbot -L CameraReaderNode -M -s -n pubisher 

PASSWORD quackquack



#  ROS Part 

* If we bulid a ROS communication we can create a file in source and make it an excutable one by 
running the following command from the root of our DTProject < chmod +x ./packages/my-package/src/my_publisher_node.py >

* Then we define a luncher by telling Docker what to run when the contener is started so that our ros project can run.

            so we add this on ./launchers/my-publisher.sh 
            "
            #!/bin/bash

            source /environment.sh

            # initialize launch file
            dt-launchfile-init

            # launch publisher
            rosrun my_package my_publisher_node.py

            # wait for app to end
            dt-launchfile-join
            "
*   This part lunch publisher node that reaquires the duckiebot up and running.
 so to make sure that our robot is ready we can execute the command < ping marsbot.local >

 Let us now re-compile our project using the command

dts devel build -H marsbot -f

and run it using the newly defined launcher (we use the flag -L/--launcher to achieve this):

dts devel run -H marsbot -L my-publisher
dts devel run -H marsbot -L camera_reader_node
dts devel run -H marsbot -L my_subscriber_node
dts devel run -H marsbot -L my_subscriber_node
dts devel run -H marsbot -L some_name_node
dts devel run -H marsbot -L wheel_control_node
dts devel run -H marsbot -L twist_control_node
















# Template: template-ros

This template provides a boilerplate repository
for developing ROS-based software in Duckietown.

**NOTE:** If you want to develop software that does not use
ROS, check out [this template](https://github.com/duckietown/template-basic).


## How to use it

### 1. Fork this repository

Use the fork button in the top-right corner of the github page to fork this template repository.


### 2. Create a new repository

Create a new repository on github.com while
specifying the newly forked template repository as
a template for your new repository.


### 3. Define dependencies

List the dependencies in the files `dependencies-apt.txt` and
`dependencies-py3.txt` (apt packages and pip packages respectively).


### 4. Place your code

Place your code in the directory `/packages/` of
your new repository.


### 5. Setup launchers

The directory `/launchers` can contain as many launchers (launching scripts)
as you want. A default launcher called `default.sh` must always be present.

If you create an executable script (i.e., a file with a valid shebang statement)
a launcher will be created for it. For example, the script file 
`/launchers/my-launcher.sh` will be available inside the Docker image as the binary
`dt-launcher-my-launcher`.

When launching a new container, you can simply provide `dt-launcher-my-launcher` as
command.
# sim_to_real_trainsfer_rl
