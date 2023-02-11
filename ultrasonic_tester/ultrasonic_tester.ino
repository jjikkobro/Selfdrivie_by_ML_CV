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

}

void loop() {
    long sdistance = ultrasonic_distance(strig, secho);
    long bdistance = ultrasonic_distance(btrig, becho);
    long fdistance = ultrasonic_distance(f1trig, f1echo);
    long rdistance = ultrasonic_distance(rtrig, recho);
    long fdistance1 = ultrasonic_distance(f2trig, f2echo); 
    Serial.print(fdistance);
    Serial.print("  ");
    Serial.print(bdistance);
    Serial.print("  ");
    Serial.print(sdistance);
    Serial.print("  ");
    Serial.print(rdistance);
    Serial.print("  ");
    Serial.print(fdistance1);
    Serial.println("  ");

}
