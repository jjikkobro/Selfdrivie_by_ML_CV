# Generate model by machine learnig

These codes for generating model to classify the steering angle by learned frame. It's based on Nvidia model and also I was inspired from Udacity self driving sequences. This is the link that I studied about self-driving : https://www.udemy.com/course/applied-deep-learningtm-the-complete-self-driving-car-course/learn/lecture/11241804?start=315#overview.

I committed sample frames for learning and I gathered 8000 frames. labels are represent left angles which are normarlized -1 to 0 and right ones are 0 to 1. Train sets and validation sets split by 8 to 2. 
I used image augmentation techniques on train sets for getting good validation accuracy. but It didn't work well. I guess that's because lack of the frames. Finally got 96% accurracy but 50% val_accurracy.
Also I tried to transfer learn with resnet-152 and VGG-16.

1. Open the code General_driving_model.py or Mission_driving_model and modify dirname that contains images and labels.csv
2. You just run the code.
3. It will generate the model.

