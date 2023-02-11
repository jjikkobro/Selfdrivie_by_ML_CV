#include <SoftwareSerial.h>
#include <ArduinoBlue.h>
#include <Car_Library.h>

const unsigned long BAUD_RATE = 9600;

//int trig = 8;
//int echo = 9;
long distance, prevDistance;

int prevThrottle = 49;
int prevSteering = 49;
int throttle, steering, throttle_value, steering_value;

int steer_motor0 = 8; 
int steer_motor1 = 9;
int wheel_motor0 = 10;
int wheel_motor1 = 11;
int wheel_motor2 = 12;
int wheel_motor3 = 13; 

ArduinoBlue phone(Serial1);

void setup() {
    Serial.begin(BAUD_RATE);
    Serial1.begin(BAUD_RATE);
//    pinMode(trig, OUTPUT);
//    pinMode(echo, INPUT);
    pinMode(steer_motor0, OUTPUT);
    pinMode(steer_motor1, OUTPUT);
    pinMode(wheel_motor0, OUTPUT);
    pinMode(wheel_motor1, OUTPUT);    
    pinMode(wheel_motor2, OUTPUT);
    pinMode(wheel_motor3, OUTPUT);   
    delay(100);
}

void loop() {
    
    throttle = phone.getThrottle();
    steering = phone.getSteering();
    //distance = ultrasonic_distance(trig, echo);

    if (throttle == 50 || steering == 50 || steering == 49 || throttle == 49){
      motor_hold(steer_motor0, steer_motor1);
      motor_hold(wheel_motor0, wheel_motor1);
      motor_hold(wheel_motor2, wheel_motor3);
    }

    else if ( throttle > 50 ) {
      throttle_value = map(throttle, 51, 99, 0, 100); 
      motor_backward(wheel_motor0, wheel_motor1, throttle_value);
      motor_forward(wheel_motor2, wheel_motor3, throttle_value);
      
      if (steering < 49) {
        steering_value = map(steering, 0, 48, 100, 0);
        motor_backward(steer_motor0, steer_motor1, steering_value);
      }
      else if (steering > 30 && steering < 65) {
        motor_hold(steer_motor0, steer_motor1);
      }
      
      else {
        steering_value = map(steering, 51, 99 ,0, 100);
        motor_forward(steer_motor0, steer_motor1, steering_value);
      }       
      if (prevThrottle != throttle || prevSteering != steering) {
        Serial.print("Throttle: "); Serial.print(throttle); Serial.print("\tSteering: "); Serial.println(steering);
        prevThrottle = throttle;
        prevSteering = steering;
      }
    }
//    else if ( throttle < 49 && steering > 30 && steering < 65) {
//      throttle_value = map(throttle, 51, 99, 0, 80); 
//      motor_hold(wheel_motor0, wheel_motor1);
//      motor_hold(wheel_motor2, wheel_motor3);
//      steering = 100;
//      if (prevThrottle != throttle || prevSteering != steering) {
//        Serial.print("Throttle: "); Serial.print(throttle); Serial.print("\tSteering: "); Serial.println(steering);
//        prevThrottle = throttle;
//        prevSteering = steering;
//      }
//    }
   
    
}
