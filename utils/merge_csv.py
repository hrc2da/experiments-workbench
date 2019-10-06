from glob import glob
import sys
import os
import csv
sys.path.append(os.getcwd())

targets_path = "/home/dev/data/distopia/team_logs/metrics/*logs.csv"
labels_path = "/home/dev/data/distopia/team_logs/metrics/*labels.csv"

target_files = glob(targets_path)
target_files.sort()
label_files = glob(labels_path)
label_files.sort()

with open("/home/dev/data/distopia/team_logs/metrics/merged_logs.csv", 'w+') as merged_targets_file:
    mtwriter = csv.writer(merged_targets_file)
    with open("/home/dev/data/distopia/team_logs/metrics/merged_logs_labels.csv", 'w+') as merged_labels_file:
        mlwriter = csv.writer(merged_labels_file)
        for fn in target_files:
            print(fn)
            with open(fn,'r') as infile:
                treader = csv.reader(infile)
                for row in treader:
                    mtwriter.writerow(row)
        for fn in label_files:
            print(fn)
            with open(fn,'r') as infile:
                lreader = csv.reader(infile)
                for row in lreader:
                    mlwriter.writerow(row)    

