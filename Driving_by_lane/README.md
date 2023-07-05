# Self-Drive by detecting lanes.

These codes are just detect the lines on the sides of the track. The road lines' color is white and the road is black. So I abstract only white color for detecting road lines.
It literally works well. But the track has some other white lines like sector for placing car that use for avoiding misson. So I tried to avoid detecting it, I had to ignore the horizontal lines.
When I organize the lines for classifying which is on the left side or right side. I decided it by slope. If it has minus slope, it is left one and + slope is right one. but there is corner, In the corner eventhough it's on the left side but has minus values of slope, So I have to divide it by center of frame.
And there is also crosswalk, but the code detects a lot of lines on crosswalk, so I made a model which detects the crosswalk. When the model detect it, stop detecting lines and getting the last angle value and it just move follow the last angle till passing it.

1. Run the Arduino code : general_driving.ino ( You have to close the arduino serial board )
2. Run the Python code : general_driving_v2.py
3. Input any value to console
4. The car will move with detecting lines

If python or arduino shows error, you just remove the cable and re-connect.

general driving : Just follow the lines.
mission driving : follow the lines and avoid the car when the ultrasonic sensor detects the car.
parking mission : park the car on the place which is between two cars.
