# Object-Detection-With-YoloV4
We used darknet here . Darknet is a open source neural network framework .
Our target object here is glasses object . This includes sunglasses and regular glasses.
First we need to collect images of our target object you can do it from stock photos or using photos from the internet .
İmages of the target object must be in .jpg format .
# Labeling
Now we can start the labeling process. We will do the tagging process on https://www.makesense.ai/ .
You can also use desktop programs like this. This site is useful because that's support output file formats like yolo,json
vog xml,csv,vgg and you can get ouput file with another formats .
# Training 
After that we can start training our own system. 
You can use open source frameworks to train your system . I will use darknet for this .
In order to use darknet, we can download the files from https://github.com/AlexeyAB/darknet .
We should use Git Installer to download these files easily . You can get it at https://git-scm.com/downloads .
After downloading them, we have to edit the darknet framework according to ourselves, I will share the edited version of the file with you .
Now we will train our system with the images we have labeled before and as a result of this training process, it will give us a file with the extension .weights. This file is the software of the trained system with mathematical functions.
In short, here the nodes of the neural network are expressed mathematically. These statements include the coordinates of the target object in the images and the problems that arise while testing it, and more .
I will tell you the details of the training process . we will train our system on linux, while doing this, you can install linux on your own computer or use a virtual linux in cloud services. Virtual linux would be a better choice. Because it requires quite a high Gpu hardware. The reason why we train our model using the GPU is that the GPU is faster than the CPU. You can use google colaboratory service as virtual machine . After performing the necessary operations on linux, you can start the training. 
I will share the details of the training process on linux with you in .ipynb format . The training process can take many hours, this may vary depending on the power of the processor you will use. 
I finished the training process in 1000 iterations, this is a sufficient number for now, but at least 10000 iterations are required for a high-level system.
When we finish the process in 1000 iterations, it will give us the trained neural network as a file with a weights extension, which is its mathematical expression.
# Testing the system
now we will run our system on python using opencv and numpy libraries. What we gained from the training process
We will use our weights file and the cfg file in the darknet files.
You can use pycharm or spyder editors via anaconda while running the system.
Now it's time to start the system . You can perform object recognition on the webcam or on a picture and video file. 
