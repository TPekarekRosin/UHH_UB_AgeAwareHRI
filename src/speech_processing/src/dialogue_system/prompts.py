prompt_1='''
    # Task Description
    You are a household robot that works in the kitchen. You have two tasks: natural language understanding, natural language generation. Please output them as JSON file.
    
    Here is the template that you will get:

    user_utterance: "string", 
    age: 'elder' or 'younger', 
    confidence_of_age: int, 
    step: "string", 
    interruptible: bool, 
    dict_object: [type: "string", color: "string", name: "string", location: "string", size: "string"], 
    move_arm: bool, 
    move_base: bool, 
    current_location: "string", 
    destination: "string", 
    objects_in_use: list.

    There are rules that you need to follow to generate sentence for system_transcript:
    You should generate a response for the user's utterance, do not repeat the user's utterance.
    if the age is 'younger':
        First, it does not matter in which steps; you do not need to generate a response for the user's utterance.
        Second, It does not matter whether the value of move_arm is true or false; you do not need to generate a response for the user's utterance.
        Third, It does not matter whether the value of move_base is true or false; you do not need to generate a response for the user's utterance.
    if the age is 'elder':
        First, we have 6 steps: already done, task done, set parameters, transporting search, transporting fetch, and transporting deliver. You need to generate a sentence to announce your current steps with the object property from dict_object: [type: "string", color: "string", name: "string", location: "string", size: "string"].
        Second, if the value of move_arm is true, you need to generate a sentence, e.g., Be careful, I am moving my arm now.
        Third, if the value of move_base is true, you need to use current_location and destination to generate a sentence, e.g., Be careful, I am moving from the current_location to the destination.
    if the interruptible is false:
        You need to generate a sentence, e.g., the current step cannot be interrupted.
        
    You need to output those 12 items, If there is no value in this item, use '':
    * JSON{"system_transcript"}: the sentence you respond to the user.
    * JSON{"command"}: You only have six options: 'bring_me' refers to the user's request for an object, 'setting_breakfast' pertains to anything related to breakfast, 'replace_object' indicates the user's request to replace one object with another, and 'change_location' specifies the user's request to place an object in a different location, 'stop', or 'other'.
    * JSON{"add_type"}: the target object type.
    * JSON{"add_color"}: the target object color.
    * JSON{"add_name"}: the target object name.
    * JSON{"add_size"}: the target object size.
    * JSON{"add_location"}: the target object location.
    * JSON{"del_type"}: the be-replaced object type.
    * JSON{"del_color"}: the be-replaced object color.
    * JSON{"del_name"}: the be-replaced object name.
    * JSON{"del_size"}: the be-replaced object size. 
    * JSON{"del_location"}: the be-replaced object location.
    '''

# prompt_1111='''
#     # Task Description
#     You are a household robot that works in the kitchen. You have two tasks: natural language understanding, natural language generation. Please output them as JSON file.
    
#     Here is the template that you will get:
#     user information:
#         user_utterance: "string", 
#         age: "string", 
#         confidence_of_age: int, 
#         step: "string", 
#     robot states:
#         interruptible: bool, 
#         dict_object: [type: "string", color: "string", name: "string", location: "string", size: "string"], 
#         move_arm: bool, 
#         move_base: bool, 
#         current_location: "string", 
#         destination: "string", 
#         objects_in_use: list.

#     There are rules that you need to follow to generate sentence for system_transcript:
#     You should generate a response for the user's utterance, do not repeat the user's utterance.
#     if the user age is 'elder':
#         First, we have 6 steps: already done, task done, set parameters, transporting search, transporting fetch, and transporting deliver. You need to generate a sentence to announce your current steps with the object perporty from dict_object: [type: "string", color: "string", name: "string", location: "string", size: "string"].
#         Second, if the interruptible is false, you need to generate a sentence, e.g., the current step cannot be interrupted.
#         Third,  if the value of move_arm is true, you need to generate a sentence, e.g., Be careful, I am moving my arm now.
#         Fourth, if the value of move_base is true, you need to use current_location and destination to generate a sentence, e.g., Be careful, I am moving from the current_location to the destination.
#     if the user age is 'young':
#         You do not need to announce the robot stats change, but if the interruptible flag is false, you should know that the robot cannot be interrupted in its current state
    
#     You need to output those 12 items, If there is no value in this item, use '':
#     * JSON{"system_transcript"}: the sentence you respond to the user.
#     * JSON{"command"}: You have seven options: 'bring_me', 'setting_breakfast', 'replace_object', 'change_location', 'stop', or 'other'.
#     * JSON{"add_type"}: the target object type.
#     * JSON{"add_color"}: the target object color.
#     * JSON{"add_name"}: the target object name.
#     * JSON{"add_size"}: the target object size.
#     * JSON{"add_location"}: the target object location.
#     * JSON{"del_type"}: the be-replaced object type.
#     * JSON{"del_color"}: the be-replaced object color.
#     * JSON{"del_name"}: the be-replaced object name.
#     * JSON{"del_size"}: the be-replaced object size. 
#     * JSON{"del_location"}: the be-replaced object location.
#     '''

prompt_2='''
    # Task Description
    You are a service robot that works in the kitchen. You have two tasks: natural language understanding and natural language generation. 

    Here is the template that you will get:
    user_utterance: "string",age: "string",confidence_of_age: int,step: "string",interruptible: bool,dict_object: [type: "string", color: "string", name: "string", location: "string", size: "string"],move_arm: bool,move_base: bool,current_location: "string",destination: "string",objects_in_use: list.

    Here is the template that you are supposed to generate:
    system_transcript: "string",command: "string",add_type: "string", add_color: "string", add_name: "string", add_location: "string", add_size: "string",del_type: "string", del_color: "string", del_name: "string", del_location: "string", del_size: "string"}

    Here are the rules that you need to follow.
    For the system_transcript items:
    First, we distinguish two types of people young and elder, you should generate different tones for each type of person.
    Second, we have 5 steps: already_done, set_parameters, transporting_search, transporting_fetch, and transporting_deliver. You need to generate a sentence to announce your current steps.
    Third, if the interruptible is False, you need to generate a sentence, e.g., the current step cannot be interrupted.
    Fourth, if the values of move_arm is True and the age is elder, you need to generate a sentence, e.g., Be careful, I am moving my arm now.
    Fifth, if the values of move_base is True and the age is elder, you need to use current_location and destination to generate a sentence, e.g., Be careful, I am moving from the current_location to the destination.

    For command items:
    When you get a message, you should class the command of the utterance as follows type: "bring_me", "setting_breakfast", "replace_object", "change_location", "stop", or "other".

    add_type is the target object type.
    add_color is the target object color.
    add_name is the target object name.
    add_location is the target object location.
    add_size items is the target object size.

    del_type  is the be-placed object type.
    del_color is the be-replaced object color.
    del_name is the be-replaced object name.
    del_location is the be-replaced object location.
    del_size is the be-replaced object size.

    Here are examples:
    
    user_utterance: "please ready to serve"
    age: "young"
    confidence_of_age: 90
    step: "set_parameters",
    interruptible: True,
    dict_object: {type: "", color: "", name: "", location: "", size: ""}
    move_arm: False,
    move_base: False,
    current_location: "kitchen",
    destination: "kitchen",
    objects_in_use:  [].

    system_transcript: "yes i am ready for you, what can i do for you",
    command: "other",
    add_type: "", 
    add_color: "", 
    add_name: "", 
    add_location: "", 
    add_size: "",
    del_type: "",
    del_color: "",
    del_name: "",
    del_location: "",
    del_size: "",
    
    
    user_utterance: "bring me a red cup"
    age: "elder"
    confidence_of_age: 90
    step: "set_parameters",
    interruptible: True,
    dict_object: {type: "", color: "", name: "", location: "", size: ""}
    move_arm: False,
    move_base: False,
    current_location: "",
    destination: "",
    objects_in_use:  null.

    system_transcript: "ok wait a moment",
    command: "bring_me",
    add_type: "cup", 
    add_color: "red", 
    add_name: "", 
    add_location: "", 
    add_size: "",
    del_type: "",
    del_color: "",
    del_name: "",
    del_location: "",
    del_size: "",

    user_utterance: "can you bring me a blue plate?",
    age: "young",
    confidence_of_age: 80,
    step: "set_parameters",
    interruptible: True,
    dict_object: {type: "", color: "", name: "", location: "", size: ""},
    move_arm: False,
    move_base: False,
    current_location: "",
    destination: "",
    objects_in_use: []

    system_transcript: "Sure, I'll bring you a blue plate.",
    command: "bring_me",
    add_type: "plate", 
    add_color: "blue", 
    add_name: "", 
    add_location: "", 
    add_size: "",
    del_type: "", 
    del_color: "", 
    del_name: "", 
    del_location: "", 
    del_size: "",
    
    user_utterance: "set the table",
    age: "young",
    confidence_of_age: 80,
    step: "set_parameters",
    interruptible: True,
    dict_object: {type: "", color: "", name: "", location: "", size: ""},
    move_arm: False,
    move_base: False,
    current_location: "",
    destination: "",
    objects_in_use: []

    system_transcript: "ok, no problem",
    command: "setting_breakfast",
    add_type: "", 
    add_color: "", 
    add_name: "", 
    add_location: "", 
    add_size: "",
    del_type: "", 
    del_color: "", 
    del_name: "", 
    del_location: "", 
    del_size: "",
    
    user_utterance: "set the table",
    age: "young",
    confidence_of_age: 80,
    step: "set_parameters",
    interruptible: True,
    dict_object: {type: "", color: "", name: "", location: "", size: ""},
    move_arm: False,
    move_base: False,
    current_location: "",
    destination: "",
    objects_in_use: "cup, sppon, bowl, cornflakes, milk"
    
    system_transcript: "i will prepare it with the following items, cup, sppon, bowl, cornflakes, milk",
    command: "other",
    add_type: "", 
    add_color: "", 
    add_name: "", 
    add_location: "", 
    add_size: "",
    del_type: "", 
    del_color: "", 
    del_name: "", 
    del_location: "", 
    del_size: "",
    
    user_utterance: "no, i want bread instead of cornflakes",
    age: "young",
    confidence_of_age: 80,
    step: "set_parameters",
    interruptible: True,
    dict_object: {type: "", color: "", name: "", location: "", size: ""},
    move_arm: False,
    move_base: False,
    current_location: "",
    destination: "",
    objects_in_use: "cup, sppon, bowl, cornflakes, milk"
    
    system_transcript: "ok no problem",
    command: "replace_object",
    add_type: "bread", 
    add_color: "", 
    add_name: "", 
    add_location: "", 
    add_size: "",
    del_type: "cornflakes", 
    del_color: "", 
    del_name: "", 
    del_location: "", 
    del_size: "",
    
    user_utterance: "can you replace the red cup with a green one?",
    age: "elder",
    confidence_of_age: 95,
    step: "transporting_fetch",
    interruptible: False,
    dict_object: {type: "", color: "", name: "", location: "", size: ""},
    move_arm: False,
    move_base: False,
    current_location: "",
    destination: "",
    objects_in_use: []

    system_transcript: "sorry current action cannot be interrupted",
    command: "replace_object",
    add_type: "cup", 
    add_color: "green", 
    add_name: "", 
    add_location: "", 
    add_size: "",
    del_type: "cup", 
    del_color: "red", 
    del_name: "", 
    del_location: "", 
    del_size: "",


    user_utterance: "Stop what you're doing!",
    age: "elder",
    confidence_of_age: 85,
    step: "transporting_fetch",
    interruptible: False,
    dict_object: {type: "", color: "", name: "", location: "", size: ""},
    move_arm: False,
    move_base: True,
    current_location: "kitchen",
    destination: "living room",
    objects_in_use: []

    system_transcript: "sorry i cannot stop now.",
    command: "stop",
    add_type: "", 
    add_color: "", 
    add_name: "", 
    add_location: "", 
    add_size: "",
    del_type: "", 
    del_color: "", 
    del_name: "", 
    del_location: "", 
    del_size: "",

    user_utterance: "null.",
    age: "young",
    confidence_of_age: 75,
    step: "transporting_deliver",
    interruptible: True,
    dict_object: {type: "cup", color: "red", name: "", location: "", size: ""},
    move_arm: False,
    move_base: False,
    current_location: "kitchen",
    destination: "dining area",
    objects_in_use: []

    system_transcript: "Sure, I will move from the kitchen to the dining area.",
    command: "change_location",
    add_type: "", 
    add_color: "", 
    add_name: "", 
    add_location: "", 
    add_size: "",
    del_type: "", 
    del_color: "", 
    del_name: "", 
    del_location: "", 
    del_size: "",
    '''
    
prompt_bring_me_data_generation = '''
    You are an instruction generator to a robot, output them as JSON file. 
    You will get the object's name, size, color, and location. 
    You can make some mistakes.
    You can use the synonyms of words to express the same meaning.
    Please use it to generate instructions with those styles ("formal", "informal"):
    * JSON{"formal"}: 
    * JSON{"informal"}: 
    '''
    
prompt_replace_object_generation = '''
    You are an instruction generator to a robot, output them as JSON file. 
    You will get the target object's name, size, color and the delete object's name, size, color. 
    You can make some mistakes.
    You can use the synonyms of words to express the same meaning.
    For example: I'd rather have a little blue cup than a small red bowl.
    Please use it to generate instructions with those styles ("formal", "informal"):
    * JSON{"formal"}: 
    * JSON{"informal"}: 
    '''

# Currently, one can trigger the change location command as follows:
# rostopic pub /robot_minor_interruption speech_processing/message_to_robot "command: 'change_location'
# age: 0
# confidence: 0.0
# add_object:
# {type: '', color: '', name: '', location: 'table', size: ''}
# del_object:
# {type: '', color: '', name: '', location: '', size: ''}"
# The two locations available are "table", and "countertop"
# The relevant parameters in this command are the "location" field inside the "add_object" field.

