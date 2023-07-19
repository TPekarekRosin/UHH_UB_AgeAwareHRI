import rospy
from speech_processing.msg import *


class DialogueSystem:
    def __init__(self):
        self.robot_state = None
        self.robot_interruptable = False
        
        self.objects_in_use = []
        
    def process_speech_input(self, transcript, age, confidence):
        # todo: define major interruptions
        if "stop" in transcript and confidence > 0.5:
            # todo: generate valid response for major interruptions
            response = message_to_robot()
            response.command = 'stop'
            response.age = age
            response.confidence = confidence
            # properties of added object
            response.add_object.append(dict_object())
            response.add_object[0].type = ''
            response.add_object[0].color = ''
            response.add_object[0].name = ''
            response.add_object[0].location = ''
            response.add_object[0].size = ''
            # properties of deleted object
            response.del_object.append(dict_object())
            response.del_object[0].type = ''
            response.del_object[0].color = ''
            response.del_object[0].name = ''
            response.del_object[0].location = ''
            response.del_object[0].size = ''

            return 'major', response

        # todo: define minor interruptions
        elif 'cup' in transcript and confidence > 0.5:
            # todo: generate valid response for minor interruptions
            response = message_to_robot()
            response.command = 'object'
            response.age = age
            response.confidence = confidence
            # properties of added object
            response.add_object.append(dict_object())
            response.add_object[0].type = 'cup'
            response.add_object[0].color = 'red'
            response.add_object[0].name = ''
            response.add_object[0].location = ''
            response.add_object[0].size = ''
            # properties of deleted object
            response.del_object.append(dict_object())
            response.del_object[0].type = ''
            response.del_object[0].color = ''
            response.del_object[0].name = ''
            response.del_object[0].location = ''
            response.del_object[0].size = ''
            return 'minor', response
        else:
            print('Command was not recognized.')
            return '', ()

    def process_robot_input(self, state, interruptable, object_info,
                            move_arm, move_base, current_loc, destination_loc):
        # todo process the message from the robot to create speech output
        self.robot_state = state
        self.robot_interruptable = interruptable
        string_for_synthesizer = f"I am {state}"
        return string_for_synthesizer

    def get_objects_in_use(self):
        # looks at state of rosparam \objects_in_use and return list of objects
        param_name = rospy.search_param('object_in_use')
        object_str = rospy.get_param(param_name)
        self.objects_in_use = object_str.split(',')

    def get_robot_states(self):
        return self.robot_state, self.robot_interruptable