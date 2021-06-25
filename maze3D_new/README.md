# Virtual Environment Overview
![Game](https://github.com/ThanasisTs/maze_RL/blob/main/pictures/maze.png)

The virtual environment is based on [this work](https://github.com/amengede/Marble-Maze)

* Control
  * The game has 2 axis of control
    * left and right arrows control the rotation of the tray around y-axis. 
    * up and down arrows control the rotation of the tray around x-axis. 

* Wall Cubes
  * Cubes with edge size: 32 pixels
  * Cube bottom Area: 1024 pixels

* Ball
  * Sphere with radius ρ<sub>2</sub>: 16 pixels
  * Ball Area: 805 pixels
  
* Goal
  * Sphere with radius ρ<sub>2</sub>: 16 pixels
  * Goal Area: 805 pixels
  
* Goal reached if the distance between the ball and the goal is less than the radius (16 pixels) of the ball.
