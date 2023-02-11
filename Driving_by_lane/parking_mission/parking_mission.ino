#include <Car_Library.h>

int f1trig = 4;
int f1echo = 5;
int btrig = 6;
int becho = 7;
int strig = 48;
int secho = 49;
int f2trig = 52;
int f2echo = 53;
int rtrig = 50;
int recho = 51;
int steer_motor0 = 8;
int steer_motor1 = 9;
int wheel_motor0 = 10;
int wheel_motor1 = 11;
int wheel_motor2 = 12;
int wheel_motor3 = 13;


void setup() {
  Serial.begin(9600);
  pinMode(f1trig, OUTPUT);
  pinMode(btrig, OUTPUT);
  pinMode(f1echo, INPUT);
  pinMode(becho, INPUT);
  pinMode(f2trig, OUTPUT);
  pinMode(f2echo, INPUT);
  pinMode(strig, OUTPUT);
  pinMode(secho, INPUT);
  pinMode(rtrig, OUTPUT);
  pinMode(recho, INPUT);
  pinMode(steer_motor0, OUTPUT);
  pinMode(steer_motor1, OUTPUT);
  pinMode(wheel_motor0, OUTPUT);
  pinMode(wheel_motor1, OUTPUT);
  pinMode(wheel_motor2, OUTPUT);
  pinMode(wheel_motor3, OUTPUT);
}

void loop() {
  while (Serial.available() > 0) {
    long sdistance = ultrasonic_distance(strig, secho);
    long bdistance = ultrasonic_distance(btrig, becho);
    long fdistance = ultrasonic_distance(f1trig, f1echo);
    Serial.print(fdistance);
    Serial.print("  ");
    Serial.print(bdistance);
    Serial.print("  ");
    Serial.print(sdistance);
    Serial.println("  ");

    motor_backward(wheel_motor0, wheel_motor1, 100);
    motor_forward(wheel_motor2, wheel_motor3, 100);
    if ( sdistance < 600 && bdistance < 600 && fdistance > 900 && fdistance < 1400) {
      while ( true ) {
        Serial.println("1stop");
        motor_hold(wheel_motor0, wheel_motor1);
        motor_hold(wheel_motor2, wheel_motor3);
        delay(1000);
        motor_backward(steer_motor0, steer_motor1, 175);
        delay(2000);
        motor_backward(wheel_motor0, wheel_motor1, 175);
        motor_forward(wheel_motor2, wheel_motor3, 175);
        long fdistance = ultrasonic_distance(f1trig, f1echo);
        delay(2000);
        Serial.println("2stop");
        motor_hold(wheel_motor0, wheel_motor1);
        motor_hold(wheel_motor2, wheel_motor3);
        motor_forward(steer_motor0, steer_motor1, 250);
        delay(1000);
        while ( true ) {
          long fdistance = ultrasonic_distance(f1trig, f1echo);
          long fdistance1 = ultrasonic_distance(f2trig, f2echo);
          motor_forward(wheel_motor0, wheel_motor1, 130);
          motor_backward(wheel_motor2, wheel_motor3, 130);
          Serial.print(fdistance);
          Serial.println(fdistance1);
          Serial.print("back");
          if ( fdistance1 < 1000 ) {
            Serial.print("3stop");
            while(true) {
              long fdistance = ultrasonic_distance(f1trig, f1echo);
              long fdistance1 = ultrasonic_distance(f2trig, f2echo);
              Serial.print(fdistance);
              Serial.print("  ");
              Serial.println(fdistance1);
              if (fdistance > 1000 && fdistance1 > 1000) {
                motor_hold(wheel_motor0, wheel_motor1);
                motor_hold(wheel_motor2, wheel_motor3);
                motor_forward(steer_motor0, steer_motor1, 100);
                delay(2000);
                escape();
              }
            }
           }
        }

      }
    }
  }
}

void escape() {
  while ( true ) {
    long fdistance = ultrasonic_distance(f1trig, f1echo);
    long rdistance = ultrasonic_distance( rtrig, recho);
    long sdistance = ultrasonic_distance(strig, secho);
    long fdistance1 = ultrasonic_distance(f2trig, f2echo);
    Serial.print(fdistance);
    Serial.print("  ");
    Serial.print(rdistance);
    Serial.print("  ");
    Serial.print(sdistance);
    Serial.print("  ");
    Serial.println(fdistance1);

    motor_backward(wheel_motor0, wheel_motor1, 200);
    motor_forward(wheel_motor2, wheel_motor3, 200);
    delay(8000);
    motor_hold(wheel_motor0, wheel_motor1);
    motor_hold(wheel_motor2, wheel_motor3);
    motor_backward(steer_motor0, steer_motor1, 50);
    Serial.print("arrange");
    delay(1000);
    motor_backward(wheel_motor0, wheel_motor1, 200);
    motor_forward(wheel_motor2, wheel_motor3, 200);
    delay(5000);
    motor_hold(wheel_motor0, wheel_motor1);
    motor_hold(wheel_motor2, wheel_motor3);
    delay(5000);
    break;
  }
}
