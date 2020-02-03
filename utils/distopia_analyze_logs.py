import distopia_human_logs_processor
import plot_distopia_metrics
import sys
#See distopia_human_logs_processor.py for log naming conventions (important)
if (len(sys.argv) != 5):
    print("USAGE: python distopia_analyze_logs.py <data path> <norm file path> <num_episodes> <episode_length>")
    exit(0)


distopia_human_logs_processor.logs_processor(sys.argv[1], sys.argv[2])
plot_distopia_metrics.plot_metrics(sys.argv[1])
plot_distopia_metrics.plot_rewards(sys.argv[1],int(sys.argv[3]),int(sys.argv[4]))
