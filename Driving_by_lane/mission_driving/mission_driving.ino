#include <Car_Library.h>
#include <SoftwareSerial.h>

int count;
long distance, bdistance;
int angle, normvalue, throttlevalue, previousA;
int steer_motor0 = 8; 
int steer_motor1 = 9;
int wheel_motor0 = 10;
int wheel_motor1 = 11;
int wheel_motor2 = 12;
int wheel_motor3 = 13; 
int trig = 50;
int echo = 51;
int btrig = 4;
int becho = 5;

void setup() {
  Serial.begin(9600);
  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(steer_motor0, OUTPUT);
  pinMode(steer_motor1, OUTPUT);
  pinMode(wheel_motor0, OUTPUT);
  pinMode(wheel_motor1, OUTPUT);    
  pinMode(wheel_motor2, OUTPUT);
  pinMode(wheel_motor3, OUTPUT);   
  pinMode(trig, OUTPUT);
  pinMode(echo, INPUT);  
  pinMode(btrig, OUTPUT);
  pinMode(becho, INPUT);
  delay(100);
}

void loop() {
  while( Serial.available() >= 1 ) {
    long readval = Serial.read();
    angle = (int)readval-48;
    distance = ultrasonic_distance(trig,echo);
    bdistance = ultrasonic_distance(btrig, becho);
    if ( bdistance < 450 ) {
      count = 1;
    }
    forward( 110 );
    if ( distance < 800 and count == 1 ) {
      hold();
      avoid( previousA, 1 );
    }
    else if ( distance < 800 ) {
      hold();
      avoid( previousA, bdistance );
    }
    if ( angle == 3 ) { 
      motor_backward( steer_motor0, steer_motor1, 200 );
      forward( 100 );
      previousA = 0;
    }
    else if ( angle == 4 ) {
      motor_forward( steer_motor0, steer_motor1, 225 );
      forward( 100 );
      previousA = 1; 
    }
    else if ( angle == 0 ) {
      motor_backward( steer_motor0, steer_motor1, 80 );
      previousA = 0;
    }
    else if ( angle == 1 ) {
      motor_forward( steer_motor0, steer_motor1, 120 );
      //forward( 120 ); 
      previousA= 1;
     
    }
    else if ( angle == 2 and previousA == 0) { 
      motor_forward( steer_motor0, steer_motor1, 170 );
      forward( 90 );
      previousA = 0;
    }
    else if ( angle == 2 and previousA == 1) { 
      motor_backward( steer_motor0, steer_motor1, 170 );
      forward( 90 );
      previousA = 1;
    }
    else if ( angle == 2 and previousA == 2) { 
      motor_hold( steer_motor0, steer_motor1 );
      forward( 90 );
      previousA = 2;
    }
    else if ( angle == 5 and previousA == 0 ) {
      motor_forward( steer_motor0, steer_motor1, 80 );
      //forward( 80 );
      hold();
      delay(2000);
      
      while (true) {
        long readval = Serial.read();
        angle = (int)readval-48;  
        if (angle == 6) {
          break;
        }
      }     
    }
    else if ( angle == 5 ) {
      hold();
      delay(1500);
      readval = Serial.read();
      readval = (int)readval-48;
      if ( readval == 0 or readval == 3 ) {
        motor_forward( steer_motor0, steer_motor1, 80);
        forward(70);
        delay(1500);
        hold();
      }
      else if ( readval == 1 or readval ==4 ) {
        motor_backward( steer_motor0, steer_motor1, 80);
        forward(70);
        delay(1500);
        hold();        
      }
      while (true) {
        long readval = Serial.read();
        angle = (int)readval-48;  
        if (angle == 6) {
          break;
        }
      }      
    }    
  }
 }




void forward( int value ) {
  motor_backward(wheel_motor0, wheel_motor1, value);
  motor_forward(wheel_motor2, wheel_motor3, value);
}

void backward( int value ) {
  motor_forward(wheel_motor0, wheel_motor1, value);
  motor_backward(wheel_motor2, wheel_motor3, value);  
}

void hold() {
   motor_hold(wheel_motor0, wheel_motor1);
   motor_hold(wheel_motor2, wheel_motor3);
}


void avoid(int A, long B ) {
  if ( B > 1000 ) {
    if ( previousA == 0 ) {
      motor_forward(steer_motor0, steer_motor1, 80);
      while(true) {
        distance = ultrasonic_distance(trig, echo);
        backward(80);
        delay(1000);
        if (distance > 1000) {
          forward(140);
          motor_backward(steer_motor0, steer_motor1, 175);
          delay(2000);
          motor_forward(steer_motor0, steer_motor1, 175);
          delay(1500);
          break;
        }
      }
    }
    else if ( previousA == 1) {
      motor_backward(steer_motor0, steer_motor1, 80);
      while(true) {
        distance = ultrasonic_distance(trig, echo);
        backward(80);
        delay(1000);
        if (distance > 1000) {
          forward(140);
          motor_backward(steer_motor0, steer_motor1, 175);
          delay(2000);
          motor_forward(steer_motor0, steer_motor1, 175);
          delay(1500);
          break;
        }
      }    
    }
  }
  else {
        if ( previousA == 0 ) {
      motor_forward(steer_motor0, steer_motor1, 80);
      while(true) {
        distance = ultrasonic_distance(trig, echo);
        backward(80);
        delay(1000);
        if (distance > 1000) {
          forward(140);
          motor_forward(steer_motor0, steer_motor1, 175);
          delay(2000);
          motor_backward(steer_motor0, steer_motor1, 175);
          delay(1500);          
          break;
        }
      }
    }
    else if ( previousA == 1) {
      motor_backward(steer_motor0, steer_motor1, 80);
      while(true) {
        distance = ultrasonic_distance(trig, echo);
        backward(80);
        delay(1000);
        if (distance > 1000) {
          forward(140);
          motor_forward(steer_motor0, steer_motor1, 175);
          delay(2000);
          motor_backward(steer_motor0, steer_motor1, 175);
          delay(1500);          
          break;
        }
      }    
    }
  }
}
