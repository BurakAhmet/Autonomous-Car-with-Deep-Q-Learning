# Autonomous Car with Deep Q-Learning
![Ekran görüntüsü 2025-01-28 175626](https://github.com/user-attachments/assets/a15f8b93-f8f0-4195-ac91-96e0ab266b07)

This repository demonstrates how to train and run a Deep Q-learning (DQN) model for controlling an autonomous car using the [AirSim](https://github.com/microsoft/AirSim/releases/download/v1.8.1-windows/AirSimNH.zip) simulation environment. The primary goal is to teach the car to navigate an environment without collisions, maximizing a reward function that encourages safe and efficient driving.
## Table of Contents
1. [Project Overview]()
2. [Installation]()
    - Requirements
3. [Usage]()
4. [How It Works?]()
    - Neural Network Architecture
    - Reward Function
5. [Results]()

## Project Overview
In this project, a deep Q-learning algorithm is employed to train a car to navigate autonomously. The car receives observations from the environment—such as camera view of the obstacles, velocity, coordinates, and sensor readings—and uses a DQN to decide the best possible steering and acceleration actions. The car trained for 1000 episodes (it took about 12 hours) to reach the end of the street and turn left without any collisions.

## Installation
### Requirements
* Python
* PyTorch 
* NumPy
* OpenCV-python
* Matplotlib
* AirSim (You can download it from https://github.com/microsoft/AirSim/releases/download/v1.8.1-windows/AirSimNH.zip)

You can download the necessary libraries from the [requirements.txt](https://github.com/BurakAhmet/Autonomous-Car-with-Deep-Q-Learning/blob/main/requirements.txt) with this command:
  ```pip install -r requirements.txt```.

## Usage
* Using an Existing Model: ```python drive.py```. Do not forget to change the model path.
* Training a model: ```python train.py```.

## How It Works
### Neural Network Architecture
![Ekran görüntüsü 2025-01-28 175451](https://github.com/user-attachments/assets/0b4c7c18-bcb1-471f-b463-5bca04e85d4f)
* The DQN model takes the current image observation from the simulation, resized to 84×84, as input. It consists of three convolutional layers that extract features from the image. The filter sizes in these layers decrease from 8×8 to 3×3, with strides adjusting from 4×4 to 1×1, and each layer is followed by ReLU activation. After the last convolutional layer, the outputs are concatenated with the current state of the simulation, represented by the X, Y , and Z coordinates, to inform the network about the vehicle’s current position. The concatenated features are passed through a fully connected layer with 512 neurons, followed by another ReLU activation. The final layer outputs five actions, representing the possible navigation decisions for the vehicle.
* The last layer outputs Q-values for each possible action.
* By selecting the action with the highest Q-value, the agent makes a decision at each step.

## Reward Function
* The reward function provides critical feedback to guide the agent's learning in the environment.
* It heavily penalizes collisions (-100 reward) by ending the episode upon collision, discouraging unsafe actions.
* It rewards the agent for reaching waypoints (+100 reward) and gives an additional bonus for completing all waypoints (+500 reward), encouraging goal-oriented behavior.
* It offers continuous feedback based on the reduction of distance to the target waypoint, rewarding movement towards the goal and penalizing moving away.
* By combining penalties and rewards, the function guides the agent to learn safe navigation while efficiently reaching its objectives.

## Results
**Reference Path**
* ![Ekran görüntüsü 2025-01-28 175517](https://github.com/user-attachments/assets/de92b2ef-1600-4106-bdf6-67d011fd58f7)
The numbered red point are represents the waypoints on the path.

**Reached Path**
* ![Ekran görüntüsü 2025-01-28 175535](https://github.com/user-attachments/assets/67a39b06-489d-4bc6-b7c2-03fa91853dd1)

**Autonomous Driving Video (accelerated x2)** 

https://github.com/user-attachments/assets/0d804259-b4cf-4437-98af-8a9bf7f66803

**Mean Max Q-value per Episode**
* ![Mean Max Q-value per Episode](https://github.com/user-attachments/assets/32d06d66-1a8f-48f5-b12e-9f49ab7742e9)

**Total Reward Per Episode**
* ![Total Rewar per Episode](https://github.com/user-attachments/assets/60267861-6661-4651-b839-871ebf5c0326)
