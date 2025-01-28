import airsim
import numpy as np
import random
from collections import deque
import torch
import time
import matplotlib.pyplot as plt
from Deep_QNetwork import Agent
from utils import (get_state_from_simulator, map_action_to_controls, compute_reward, save_checkpoint, compute_q_values,
                   load_checkpoint)

# Set random seeds for reproducibility to ensure consistent results across runs
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


# Main execution starts here
if __name__ == "__main__":
    # Create an AirSim client instance for controlling the car in the simulator
    client = airsim.CarClient()
    client.confirmConnection()     # Confirm the connection to the simulator
    client.enableApiControl(True)  # Enable API control over the car
    car_controls = airsim.CarControls()  # Initialize car controls

    # Define the number of possible actions the agent can take
    # Actions: Steer left, go straight, steer right, slow down, speed up (total of 5 actions)
    action_size = 5

    # Create the agent with the defined action size
    agent = Agent(action_size)

    # Number of episodes to run during training
    episodes = 1000

    # List to store the total reward obtained in each episode
    reward_per_episode = []

    # List to store mean max Q-values per episode
    q_values_per_episode = []

    # Early stopping parameters to prevent overfitting or unnecessary training
    best_total_reward = float('-inf')  # Initialize the best total reward
    patience = 1000  # Number of episodes to wait for improvement before stopping
    patience_counter = 0  # Counter to keep track of episodes without improvement

    # Optionally load from a checkpoint to resume training
    # Uncomment the following line if you have a checkpoint file
    # start_episode = load_checkpoint('checkpoint.pth', agent)
    start_episode = 0  # Start from the first episode if not loading from a checkpoint

    # Define waypoints for the car to follow (Replace with your own coordinates)
    waypoints = [
        # Starting Street
        airsim.Vector3r(-3.8146975356312396e-08, 7.445784285664558e-05, -0.5857376456260681),  # Starting point
        airsim.Vector3r(21.836427688598633, -0.024445464834570885, -0.5837180614471436),  # White car
        airsim.Vector3r(51.68717575073242, -0.5642141103744507, -0.584981381893158),  # Red car
        airsim.Vector3r(80.388427734375, -1.1560953855514526, -0.5853434801101685),  # Near end of the street
        airsim.Vector3r(119.025634765625, -1.2211841344833374, -0.5852082371711731),  # End of the street

        # First Right Turn Street
        # airsim.Vector3r(128.8560333251953, 10.253881454467773, -0.5845767259597778),           # Right 1
        # airsim.Vector3r(128.94061279296875, 30.54828453063965, -0.5842372179031372),           # Right 2
        # airsim.Vector3r(128.113037109375, 51.592010498046875, -0.5817304253578186),            # Red Car
        # airsim.Vector3r(128.45367431640625, 80.0430679321289, -0.5816816687583923),            # Garden Door
        # airsim.Vector3r(127.05535125732422, 118.08218383789062, -0.5941388010978699),          # End of the street

        # Second Right Turn Street
        # airsim.Vector3r(111.99571228027344, 126.69385528564453, -0.5815831422805786),          # Start of the street/Garage Door
        # airsim.Vector3r(86.76616668701172, 126.91148376464844, -0.5817050337791443),           # White Car

        # First Left Turn Street
        airsim.Vector3r(128.579345703125, -12.327738761901855, -0.58383709192276),
        airsim.Vector3r(129.1442108154297, -26.571720123291016, -0.5849749445915222),
        airsim.Vector3r(128.38992309570312, -46.184783935546875, -0.5853114724159241),
        airsim.Vector3r(128.2161407470703, -66.68011474609375, -0.5851855278015137),
    ]

    # Training loop over episodes
    for e in range(start_episode, episodes):
        # Reset the simulator environment and the state stack for the new episode
        client.reset()  # Reset the car to its original state
        client.enableApiControl(True)  # Ensure API control is enabled
        time.sleep(0.1)  # Brief pause to allow the simulator to reset
        car_controls = airsim.CarControls()  # Reset car controls

        # Initialize the index of the current waypoint
        current_waypoint_index = 0

        # Get the initial state from the simulator: image and relative position to the first waypoint
        state_image, relative_position = get_state_from_simulator(
            client, waypoints[current_waypoint_index])

        # Initialize a deque (double-ended queue) to store the last 4 state images (frames)
        state_stack = deque(maxlen=4)
        for _ in range(4):
            # For the initial state, duplicate the same image to fill the stack
            state_stack.append(state_image)

        # Convert the deque of images into a numpy array to create a stacked state
        # Shape will be (4, 84, 84): 4 grayscale images of size 84x84
        stacked_state = np.array(state_stack)

        total_reward = 0  # Initialize the total reward for the current episode

        # Initialize previous_distance
        car_state = client.getCarState()
        car_position = car_state.kinematics_estimated.position
        target_waypoint = waypoints[current_waypoint_index]

        previous_distance = np.sqrt(
            (car_position.x_val - target_waypoint.x_val) ** 2 +
            (car_position.y_val - target_waypoint.y_val) ** 2
        )

        # Initialize list to store Q-values per time step in this episode
        q_values_in_episode = []

        # Time steps within an episode (maximum of 500 steps)
        for time_step in range(500):
            # The agent selects an action based on the current stacked state and relative position
            action = agent.act(stacked_state, relative_position)

            # Compute Q-values and store
            q_values = compute_q_values(agent, stacked_state, relative_position)
            q_values_in_episode.append(q_values)

            # Map the action index to actual car control commands (throttle and steering)
            car_controls = map_action_to_controls(action)

            # Send the control commands to the car in the simulator
            client.setCarControls(car_controls)

            # Get the next state from the simulator after performing the action
            next_state_image, next_relative_position = get_state_from_simulator(
                client, waypoints[current_waypoint_index])

            # Optional: Test image capture for DEBUGGING purposes
            # Uncomment to display the image
            # cv2.imshow('Grayscale Image', next_state_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # Update the state stack with the new image
            state_stack.append(next_state_image)
            # Create the next stacked state from the updated state stack
            stacked_next_state = np.array(state_stack)

            # Compute the reward for the action, check if the episode is done,
            # and update the current waypoint index if necessary
            # Compute reward using the previous_distance
            reward, done, current_waypoint_index, current_distance = compute_reward(
                client, waypoints, current_waypoint_index, previous_distance)
            total_reward += reward

            # Update previous_distance for the next step
            previous_distance = current_distance

            # Store the experience (state transition) in the agent's replay memory
            agent.remember(
                stacked_state,            # Current state
                relative_position,        # Relative position to the waypoint
                action,                   # Action taken
                reward,                   # Reward received
                stacked_next_state,       # Next state
                next_relative_position,   # Next relative position
                done                      # Whether the episode is done
            )

            # Train the agent with experiences sampled from the replay memory
            agent.replay()

            # Update the current state and relative position for the next time step
            stacked_state = stacked_next_state
            relative_position = next_relative_position

            # Check if the episode is done (e.g., collision, waypoint reached, or max time steps)
            if done or time_step == 499:
                # Print statistics for the current episode
                print(f"Episode {e + 1}/{episodes} - Time Steps: {time_step + 1}, "
                      f"Total Reward: {total_reward}, "
                      f"Epsilon: {agent.epsilon:.2f}")
                break  # Exit the time steps loop to start a new episode

        # After the episode ends, store the total reward
        reward_per_episode.append(total_reward)

        # Compute mean max Q-value per episode and store it
        q_values_in_episode = np.array(q_values_in_episode)  # Shape: (num_steps, action_size)
        max_q_values_per_step = np.max(q_values_in_episode, axis=1)  # Shape: (num_steps,)
        mean_max_q_value_episode = np.mean(max_q_values_per_step)
        q_values_per_episode.append(mean_max_q_value_episode)

        # Check if the agent has achieved a new best total reward
        if total_reward > best_total_reward:
            best_total_reward = total_reward  # Update the best total reward
            patience_counter = 0  # Reset the patience counter
            # Save the best model's parameters to a file
            torch.save(agent.model.state_dict(), 'best_dqn_model.pth')
            print(f"Episode {e + 1}: New best total reward: "
                  f"{best_total_reward:.2f}. Model saved.")
        else:
            # No improvement, increment the patience counter
            patience_counter += 1

        # Early stopping: check if the patience limit has been reached
        if patience_counter >= patience:
            print(f"No improvement for {patience} episodes. Early stopping.")
            break  # Exit the episodes loop to end training

        # Update the target network every fixed number of episodes to stabilize training
        if e % 5 == 0:
            agent.update_target_model()

        # Save a checkpoint of the model and training state every 10 episodes
        if e % 100 == 0:
            save_checkpoint({
                'episode': e + 1,
                'model_state_dict': agent.model.state_dict(),
                'target_model_state_dict': agent.target_model.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon,
            }, filename=f'checkpoint_episode_{e + 1}.pth.tar')

    # After all episodes are completed, plot the rewards per episode to visualize training progress
    plt.plot(reward_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.savefig('rewards_per_episode.png')
    plt.show()

    # Plot the mean max Q-value per episode
    plt.figure()
    plt.plot(q_values_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Mean Max Q-value')
    plt.title('Mean Max Q-value per Episode')
    plt.savefig('mean_max_q_values.png')
    plt.show()
