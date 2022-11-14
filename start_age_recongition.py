import os
import torch
import logging

import pyaudio as pa
import stomp
import yaml

from age_recognition.live_model import ASRLiveModel

# Init command line logging
logging.basicConfig(level='INFO', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt="%m/%d/%y %H:%M:%S")
logger = logging.getLogger('General-Log')

# load config for asr model
folder, _ = os.path.split(__file__)
filepath = os.path.join(os.path.dirname(folder), 'configs', 'live_model_config.yaml')
with open(filepath) as f:
    config = yaml.safe_load(f)

# Set base path (current working directory)
base_path = os.getcwd()

# Set processing device
device = torch.device('cuda:' + str(config['gpu'])) if torch.cuda.is_available() else torch.device('cpu')

sample_rate = config["sample_rate"]
model_name = config["model_name"]
command_list = "./configs/sentences.txt"

# online inference
p = pa.PyAudio()
logger.info('Available audio input devices:')
input_devices = []
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if dev.get('maxInputChannels'):
        input_devices.append(i)
        print(i, dev.get('name'))

if len(input_devices):
    dev_idx = -2
    while dev_idx not in input_devices:
        logger.info('Please type input device ID:')
        dev_idx = int(input())

    device = p.get_device_info_by_index(dev_idx)

    logger.info("Loading ASR Model...")
    asr = ASRLiveModel(device.get('name'), **config)
    asr.start()

    try:
        while True:
            text, command, confidence, age_estimation, sample_length, inference_time = asr.get_last_text()
            logger.info(f"Sample length: {sample_length:.3f}s"
                        + f"\tInference time: {inference_time:.3f}s"
                        + f"\tAge Estimation: {age_estimation:.3f}"
                        + f"\tHeard: '{text}'")
            logger.info("Assume command: '", command, "', with confidence of ", confidence)
    except KeyboardInterrupt:
        asr.stop()
else:
    logger.info('ERROR: No audio input device found.')
