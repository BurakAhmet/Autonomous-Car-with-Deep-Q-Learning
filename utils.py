import airsim
import torch
import numpy as np
import cv2


def get_state_from_simulator(client, current_waypoint):
    """
    Retrieves the current state from the AirSim simulator, including a processed image and the normalized relative position to the current waypoint.

    Parameters:
    - client: AirSim client used to interact with the simulator.
    - current_waypoint: The current target waypoint position.

    Returns:
    - img_normalized: A normalized 84x84 grayscale image from the simulator's camera.
    - relative_position_normalized: A normalized vector representing the relative position to the current waypoint.
    """
    # Request an image from the simulator's camera with name "0".
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
    ])
    response = responses[0]

    # Convert the image data from a byte buffer to a 1D numpy array.
    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)

    # Handle cases where the image might not be returned correctly.
    if img1d.size != response.height * response.width * 3:
        # If the image size is incorrect, create a zeroed array with the expected size.
        img1d = np.zeros(
            (response.height * response.width * 3,), dtype=np.uint8)

    # Reshape the 1D image array to a 3D array (height x width x channels).
    img_rgb = img1d.reshape(response.height, response.width, 3)

    # Convert the RGB image to grayscale.
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    # Resize the image to 84x84 pixels to ensure a consistent input size.
    img_gray = cv2.resize(img_gray, (84, 84))

    # Normalize pixel values to the range [0, 1].
    img_normalized = img_gray.astype(np.float32) / 255.0

    # Get the car's current state from the simulator.
    car_state = client.getCarState()

    # Extract the car's current position.
    car_position = car_state.kinematics_estimated.position

    # Compute the relative position vector to the current waypoint.
    relative_position = np.array([
        current_waypoint.x_val - car_position.x_val,
        current_waypoint.y_val - car_position.y_val,
        current_waypoint.z_val - car_position.z_val
    ], dtype=np.float32)

    # Compute the norm (magnitude) of the relative position vector.
    norm = np.linalg.norm(relative_position) + 1e-5  # Add a small value to avoid division by zero.

    # Normalize the relative position vector.
    relative_position_normalized = relative_position / norm

    # Return the processed image and the normalized relative position.
    return img_normalized, relative_position_normalized


def map_action_to_controls(action):
    """
    Maps a numerical action to specific car control commands.

    Parameters:
    - action: An integer representing the action to take.

    Returns:
    - car_controls: An AirSim CarControls object with the specified throttle and steering commands.
    """
    # Initialize the car controls object.
    car_controls = airsim.CarControls()

    # Map the action to car controls.
    if action == 0:
        # Action 0: Steer left at medium speed.
        car_controls.throttle = 0.5
        car_controls.steering = -0.5
    elif action == 1:
        # Action 1: Go straight at medium speed.
        car_controls.throttle = 0.5
        car_controls.steering = 0.0
    elif action == 2:
        # Action 2: Steer right at medium speed.
        car_controls.throttle = 0.5
        car_controls.steering = 0.5
    elif action == 3:
        # Action 3: Slow down and go straight.
        car_controls.throttle = 0.3
        car_controls.steering = 0.0
    elif action == 4:
        # Action 4: Speed up and go straight.
        car_controls.throttle = 0.8
        car_controls.steering = 0.0

    # Return the configured car controls.
    return car_controls


def compute_reward(client, waypoints, current_waypoint_index, previous_distance):
    """
    Computes the reward for the agent based on its current state, collisions, and progress toward waypoints.

    Parameters:
    - client: AirSim client used to interact with the simulator.
    - waypoints: List of waypoints the agent needs to reach.
    - current_waypoint_index: Index of the current waypoint the agent is aiming for.

    Returns:
    - reward: A float value representing the reward.
    - done: A boolean indicating whether the episode is done.
    - current_waypoint_index: Updated index of the current waypoint.
    - distance: The distance to the current waypoint.
    """
    # Check for any collision.
    collision_info = client.simGetCollisionInfo()
    car_state = client.getCarState()
    car_position = car_state.kinematics_estimated.position

    if collision_info.has_collided:
        return -100.0, True, current_waypoint_index, None  # End episode on collision

    # Get the current waypoint
    target_waypoint = waypoints[current_waypoint_index]
    # Calculate distance to the waypoint
    distance = np.sqrt(
        (car_position.x_val - target_waypoint.x_val) ** 2 +
        (car_position.y_val - target_waypoint.y_val) ** 2
    )

    # Define a threshold to consider the waypoint "reached"
    threshold_distance = 5.0

    # Check if waypoint is reached
    if distance < threshold_distance:
        # Move to the next waypoint
        current_waypoint_index += 1
        reward = 100.0  # Reward for reaching the waypoint
        done = False

        # Check if all waypoints are completed
        if current_waypoint_index >= len(waypoints):
            reward += 500.0  # Extra reward for completing the path
            done = True
            return reward, done, current_waypoint_index, distance
    else:
        # Reward based on reduction in distance
        if previous_distance is not None:
            reward = previous_distance - distance
        else:
            reward = 0.0  # No reward on the first step
        done = False

    return reward, done, current_waypoint_index, distance


def compute_q_values(agent, state, relative_position):
    """
    Computes Q-values for a given state and relative position using the agent's policy network.

    By plotting the mean max Q-value per episode,
    you can observe how the agent's confidence in its decisions evolves over time.
    Typically, as the agent learns, the Q-values should increase, reflecting better estimations of the expected rewards.

    Parameters:
    - agent: The agent object containing the policy network.
    - state (ndarray): The current state of the environment.
    - relative_position (ndarray): The relative position vector.

    Returns:
    - q_values (ndarray): An array of Q-values for each action.
    """

    with torch.no_grad():
        # Prepare state and relative position tensors
        state_tensor = torch.from_numpy(state).unsqueeze(0).float()  # Add batch dimension
        relative_position_tensor = torch.from_numpy(relative_position).unsqueeze(0).float()
        if torch.cuda.is_available():
            state_tensor = state_tensor.cuda()
            relative_position_tensor = relative_position_tensor.cuda()
        # Set model to evaluation mode
        agent.model.eval()
        # Compute Q-values
        q_values = agent.model(state_tensor, relative_position_tensor)
        # Return Q-values as numpy array
        return q_values.cpu().numpy().flatten()


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    Saves the current state of the training process to a checkpoint file.

    Parameters:
    - state: A dictionary containing the model's state_dict, optimizer state, episode number, etc.
    - filename: The filename where the checkpoint will be saved.
    """
    # Save the state dictionary to the specified file.
    torch.save(state, filename)


def load_checkpoint(filename, agent):
    """
    Loads the training state from a checkpoint file and updates the agent's parameters.

    Parameters:
    - filename: The filename from which the checkpoint will be loaded.
    - agent: The agent object whose parameters are to be updated.

    Returns:
    - start_episode: The episode number from which to resume training.
    """
    # Load the checkpoint from the file.
    checkpoint = torch.load(filename)

    # Load model parameters into the agent's model and target model.
    agent.model.load_state_dict(checkpoint['model_state_dict'])
    agent.target_model.load_state_dict(checkpoint['target_model_state_dict'])

    # Load the optimizer state.
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load the epsilon value for the exploration rate.
    agent.epsilon = checkpoint['epsilon']

    # Get the episode number to resume training from.
    start_episode = checkpoint['episode']

    # Return the episode number to start from.
    return start_episode