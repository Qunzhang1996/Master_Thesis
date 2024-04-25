import pandas as pd
import matplotlib.pyplot as plt

def plot_vehicle_trajectory(csv_path, iteration, vehicle_id, controller):
    # Load the data
    df = pd.read_csv(csv_path)

    # Filter data for the specific iteration, vehicle ID, and controller
    df_filtered = df[(df['Iteration'] == iteration) & (df['Vehicle ID'] == vehicle_id) ]

    # Extract traffic_x and traffic_y for the filtered data
    traffic_x = df_filtered[df_filtered['Feature'] == 'traffic_x'][controller]
    print(traffic_x)
    traffic_y = df_filtered[df_filtered['Feature'] == 'traffic_y'][controller]

    # Plotting the trajectory
    plt.figure(figsize=(10, 6))
    plt.plot(traffic_x, traffic_y, marker='o', linestyle='-', color='b')
    plt.title(f'Trajectory for Iteration {iteration}, Vehicle ID {vehicle_id}, and Controller {controller}')
    plt.xlabel('Traffic X')
    plt.ylabel('Traffic Y')
    plt.grid(True)
    plt.show()

# Example usage
if __name__ == "__main__":
    csv_path = r'C:\Users\A490243\Desktop\Master_Thesis\param_logs.csv'
    iteration = 0  # Specify the iteration number
    vehicle_id = 1  # Specify the vehicle ID
    #controller = doLeft
    controller = 'doLeft'  # Specify the controller
    plot_vehicle_trajectory(csv_path, iteration, vehicle_id, controller)
