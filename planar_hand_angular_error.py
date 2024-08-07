#%%
def compute_angular_error(angle, goal_angle):
    delta = 360 * (goal_angle - angle) / (2 * np.pi)
    delta_wraped = delta - (delta // 360) * 360
    if delta_wraped > 180:
        delta_wraped -= 360
    elif delta_wraped < -180:
        delta_wraped += 360
    angular_error = np.abs(delta_wraped)
    return angular_error

def compute_minimal_angular_error(prob_rrt):
    rrt_params = prob_rrt.rrt_params
    max_size = rrt_params.max_size
    goal_angle = rrt_params.goal[2]
    angular_errors = np.array([compute_angular_error(prob_rrt.q_matrix[i,2], goal_angle) for i in range(max_size)])
    return np.min(angular_errors)


def compute_closest_node_angular_error(prob_rrt):
    rrt_params = prob_rrt.rrt_params
    d_batch = prob_rrt.calc_distance_batch(rrt_params.goal)
    node_id_closest = np.argmin(d_batch)
    goal_angle = rrt_params.goal[2]
    angular_error = compute_angular_error(prob_rrt.q_matrix[node_id_closest,2], goal_angle)
    return angular_error


closest_node_angular_error = compute_closest_node_angular_error(prob_rrt)
minimal_angular_error = compute_minimal_angular_error(prob_rrt)

print("closest node angular error", closest_node_angular_error)
print("minimal angular error", minimal_angular_error)


#%%
# we benchmark the performance of the RRT search on 4 tasks: planar hand rotate the ball by 45, 90, 135, and 180 degrees 
# quasidynamic assumption, gravity vector aligned with ball rotation vector (i.e. "not pointing downward")
# we run 1400 iterations per RRT search (this corresponds to 1 min of compute per search on the bdai laptop)
# we run each task 50 times and report the average and std angular error for the ball.
# minimal_angular_error is the error of the state with minimal error in the tree
# closest_node_angular_error is the erro of the state considered closets to the goal

#%%
# goal_angles = np.array([45, 90, 135, 180]) / 360 * 2 * np.pi
# num_samples = 50
# num_iterations = 1400
goal_angles = np.array([45]) / 360 * 2 * np.pi
num_samples = 1
num_iterations = 14


#%%
rrt_params.max_size = num_iterations

minimal_angular_errors = np.zeros((len(goal_angles), num_samples))
closest_node_angular_errors = np.zeros((len(goal_angles), num_samples))

for task_idx, goal_angle in enumerate(goal_angles):
    for sample_idx in range(num_samples):
        rrt_params.goal[2] = goal_angle
        prob_rrt = IrsRrtProjection(rrt_params, contact_sampler, q_sim, q_sim_py)
        prob_rrt.iterate()
        minimal_angular_errors[task_idx, sample_idx] = compute_minimal_angular_error(prob_rrt)
        closest_node_angular_errors[task_idx, sample_idx] = compute_closest_node_angular_error(prob_rrt)

print("minimal_angular_errors", minimal_angular_errors)
print("closest_node_angular_errors", closest_node_angular_errors)


#%%
# Convert matrices to DataFrame
df_minimal_angular_errors = pd.DataFrame(minimal_angular_errors)
df_closest_node_angular_errors = pd.DataFrame(closest_node_angular_errors)

# Insert goal_angles as the first column
df_minimal_angular_errors.insert(0, 'goal_angle', goal_angles * 360 / (2 * np.pi))
df_closest_node_angular_errors.insert(0, 'goal_angle', goal_angles * 360 / (2 * np.pi))

# Save DataFrames to CSV files

# Ensure the directory exists
benchmark_dir = "benchmark_data"
os.makedirs(benchmark_dir, exist_ok=True)

df_minimal_angular_errors.to_csv("benchmark_data/minimal_angular_errors.csv", index=False)
df_closest_node_angular_errors.to_csv("benchmark_data/closest_node_angular_errors.csv", index=False)