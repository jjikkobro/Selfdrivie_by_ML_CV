#include <Car_Library.h>
#include <SoftwareSerial.h>

int previousA;
int angle, normvalue, throttlevalue;
int steer_motor0 = 8; 
int steer_motor1 = 9;
int wheel_motor0 = 10;
int wheel_motor1 = 11;
int wheel_motor2 = 12;
int wheel_motor3 = 13; 


void setup() {
  Serial.begin(9600);
  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(steer_motor0, OUTPUT);
  pinMode(steer_motor1, OUTPUT);
  pinMode(wheel_motor0, OUTPUT);
  pinMode(wheel_motor1, OUTPUT);    
  pinMode(wheel_motor2, OUTPUT);
  pinMode(wheel_motor3, OUTPUT);   
  delay(100);
}

void loop() {
  while( Serial.available() >= 1 ) {
    long readval = Serial.read();
    angle = (int)readval-48;
    forward( 110 );
    if ( angle == 3 ) { 
      motor_backward( steer_motor0, steer_motor1, 200 );
      //forward( 100 );
      previousA = 0;
    }
     else if ( angle == 4 ) {
      motor_forward( steer_motor0, steer_motor1, 225 );
      //forward( 100 );
      previousA = 1; 
    }
    else if ( angle == 0 ) {
      motor_backward( steer_motor0, steer_motor1, 120 );
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

    if ( angle == 5 and previousA == 0 ) {
      hold();
      delay(500);
      motor_hold( steer_motor0, steer_motor1 );
      forward( 80 );
      delay(1500);
    }
    else if ( angle == 5 and previousA == 1 ) {
      hold();
      motor_hold( steer_motor0, steer_motor1 );
      forward( 80 );
      delay(1500);
    }
    else if ( angle == 5 and previousA == 2 ) {
      motor_hold( steer_motor0, steer_motor1 );
      //forward( 120 );
      previousA = 2;
    }    
  }
 }


void forward( int value ) {
  motor_backward(wheel_motor0, wheel_motor1, value);
  motor_forward(wheel_motor2, wheel_motor3, value);
}

void hold() {
   motor_hold(wheel_motor0, wheel_motor1);
   motor_hold(wheel_motor2, wheel_motor3);
}
