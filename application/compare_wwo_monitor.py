"""
Compare the performance of agents with and without the plan monitor integrated.
Get the instances to be compared from the csv file. (stats/flawed_trajs/{config}.csv). The vanilla trajectories are stored in data/{agent}/trajectories/{mapped model}/
The monitored trajectories are stored in application/{agent}/trajectories/{config}/exp-{run_id}/{model}/
Compute the length of the trajectory, success rate (from report/), oscillation rate, and plan compliance rate.
Example usage:
Agent default to SWE-agent now. run_id default to 1.
python compare_wwo_monitor.py --config oscillation --run_id 1
"""