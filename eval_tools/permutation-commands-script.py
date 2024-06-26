import itertools
import yaml

# Define the elements for permutation
commands_minor = ['bring_me', 'replace_object']
commands_major = ['stop']
objects = [
    {'type': 'milk', 'color': 'blue', 'size': 'Normal'},
    {'type': 'milk', 'color': 'red', 'size': 'Big'},
    {'type': 'bowl', 'color': 'white', 'size': 'Normal'},
    {'type': 'cereal', 'color': 'green', 'size': 'Normal'},
    {'type': 'spoon', 'color': 'blue', 'size': 'Normal'},
    {'type': 'cup', 'color': 'white', 'size': 'Normal'}
]
ages = [0, 1]

# Create permutations for minor commands (bring_me and replace_object) with age variations
permutations_minor_bring_me = [
    {
        'command': 'bring_me',
        'age': age,
        'confidence': 0.0,
        'add_object': [obj],
        'del_object': []
    }
    for obj in objects
    for age in ages
]

permutations_minor_replace_object = [
    {
        'command': 'replace_object',
        'age': age,
        'confidence': 0.0,
        'add_object': [add_obj],
        'del_object': [del_obj]
    }
    for add_obj, del_obj in itertools.permutations(objects, 2)
    for age in ages
]

# Major command stop with age variations
permutations_major = [
    {
        'command': 'stop',
        'age': age,
        'confidence': 0.0,
        'add_object': [],
        'del_object': []
    }
    for age in ages
]

# Combine all permutations
all_permutations = permutations_minor_bring_me + permutations_minor_replace_object + permutations_major

# Convert permutations to ROS message publications
ros_messages = [
    f"rostopic pub /robot_minor_interruption speech_processing/message_to_robot \"{yaml.dump(perm, default_flow_style=True)}\""
    for perm in permutations_minor_bring_me + permutations_minor_replace_object
] + [
    f"rostopic pub /robot_major_interruption speech_processing/message_to_robot \"{yaml.dump(perm, default_flow_style=True)}\""
    for perm in permutations_major
]

# Save to a markdown format
markdown_content = '\n\n'.join(f'```yaml\n{msg}\n```' for msg in ros_messages)

with open("ros_messages_permutations.md", "w") as file:
    file.write(markdown_content)

# Print ROS messages for copy-pasting
print(markdown_content)
