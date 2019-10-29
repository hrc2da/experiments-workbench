import distopia_human_logs_processor
import plot_distopia_metrics
import sys
#See distopia_human_logs_processor.py for log naming conventions (important)
if (len(sys.argv) != 3):
    print("USAGE: python distopia_analyze_logs.py <data path> <norm file path>")
    exit(0)

distopia_human_logs_processor.logs_processor(sys.argv[1], sys.argv[2])
plot_distopia_metrics.plot_metrics(sys.argv[1])
