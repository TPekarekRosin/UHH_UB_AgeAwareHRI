import json
import os
from collections import OrderedDict
from datetime import datetime
now = datetime.now()

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def combine_statistics_from_directory(directory_path):
    combined_statistics = {}

    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            statistics = read_json_file(file_path)
            for key, value in statistics.items():
                if key in combined_statistics:
                    combined_statistics[key] += value
                else:
                    combined_statistics[key] = value

    # Calculate additional statistics
    total_commands = combined_statistics.get('total_commands', 0)
    objects_replaced = combined_statistics.get('objects_replaced', 0)
    objects_not_correct = combined_statistics.get('objects_not_correct', 0)
    ignored_commands = combined_statistics.get('ignored_commands', 0)
    changed_locations = combined_statistics.get('changed_locations', 0)

    # Calculate ignored commands rate
    if total_commands > 0:
        combined_statistics['ignored_commands_rate'] = (ignored_commands / total_commands) * 100
    else:
        combined_statistics['ignored_commands_rate'] = 0.0

    # Calculate failed/lost commands
    failed_lost_commands = total_commands - (
            objects_replaced + objects_not_correct + ignored_commands + changed_locations)
    combined_statistics['failed/lost_commands'] = failed_lost_commands

    return combined_statistics


def write_json_file(data, file_path):
    # Specify the desired order of keys
    ordered_data = OrderedDict()
    ordered_data['total_commands'] = data.get('total_commands', 0)
    ordered_data['objects_replaced'] = data.get('objects_replaced', 0)
    ordered_data['changed_locations'] = data.get('changed_locations', 0)
    ordered_data['objects_not_correct'] = data.get('objects_not_correct', 0)
    ordered_data['ignored_commands'] = data.get('ignored_commands', 0)
    ordered_data['lost_commands'] = data.get('failed/lost_commands', 0)
    ordered_data['failure_success_rate'] = data.get('failure_success_rate', 0.0)
    ordered_data['ignored_commands_rate'] = data.get('ignored_commands_rate', 0.0)

    with open(file_path, 'w') as file:
        json.dump(ordered_data, file, indent=4)
short_str = now.strftime("-%d-%m-%y-%H:%M")

directory_path = '../robot_logs/'  # specify your directory path here
output_file = 'robot_combined_statistics' + short_str + '.json'
print(f"Combined robot statistics")
combined_statistics = combine_statistics_from_directory(directory_path)
write_json_file(combined_statistics, output_file)
print(f"Combined statistics have been written to {output_file}")
