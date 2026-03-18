const int ENA = 5; //speed for motor A
const int IN1 = 8; //direction for left motor
const int IN2 = 9; 

const int ENB = 6; //same as motor A
const int IN3 = 10; //direction for right motor
const int IN4 = 11;

char data;
unsigned long lastCommandTime = 0;

int forward_speed = 180;
int turn_speed = 120;

bool is_stop = true;

void setup() {
  // put your setup code here, to run once:
  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);

  pinMode(ENB, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  Serial.begin(9600);
  stopCar();
}

void forward(){
  analogWrite(ENA, forward_speed);
  analogWrite(ENB, forward_speed);

  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW); //move forward

  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);

  is_stop = false;
}

void backward(){
  analogWrite(ENA, forward_speed);
  analogWrite(ENB, forward_speed);

  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH); //move backward

  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);

  is_stop = false;
}

void stopCar(){
  analogWrite(ENA, 0);
  analogWrite(ENB, 0);

  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW); //stop

  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);

  is_stop = true;
}

void turn_left(){
  analogWrite(ENA, turn_speed);
  analogWrite(ENB, turn_speed);

  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH); //backward

  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW); //forward

  is_stop = false;
}

void turn_right(){
  analogWrite(ENA, turn_speed);
  analogWrite(ENB, turn_speed);

  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW); //forward

  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH); //backward

  is_stop = false;
}

void loop() {
  // put your main code here, to run repeatedly:
  /*
  Test if the car can works
  forward();
  delay(2000);

  stopCar();
  delay(1000);

  backward();
  delay(2000);

  stopCar();
  delay(1000);

  turn_left();
  delay(1500);

  stopCar();
  delay(1000);

  turn_right();
  delay(1500);

  stopCar();
  delay(2000);
  */
  if (Serial.available() > 0){
    data = Serial.read();
    Serial.print(data);
    Serial.print("\n");

    if (data == '\n' || data == '\r') {
      return;
    }

    lastCommandTime = millis();

    if (data == 'F'){
      forward();
    }

    else if (data == 'B'){
      backward();
    }
    
    else if (data == 'S'){
      stopCar();
    }
    
    else if (data == 'L'){
      turn_left();
    }

    else if (data == 'R'){
      turn_right();
    }
  }

  if (!is_stop && millis() - lastCommandTime >= 10000){
      stopCar();
      Serial.print("Auto stop: no command received for 10 seconds.");
    }
}
