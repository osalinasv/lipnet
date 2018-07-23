import os

from dotenv import load_dotenv

ENV_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), '.env'))
load_dotenv(dotenv_path=ENV_PATH)

FRAME_COUNT    = int(os.getenv('FRAME_COUNT'))
IMAGE_WIDTH    = int(os.getenv('IMAGE_WIDTH'))
IMAGE_HEIGHT   = int(os.getenv('IMAGE_HEIGHT'))
IMAGE_CHANNELS = int(os.getenv('IMAGE_CHANNELS'))

MAX_STRING     = int(os.getenv('MAX_STRING'))
OUTPUT_SIZE    = int(os.getenv('OUTPUT_SIZE'))

MINIBATCH_SIZE = int(os.getenv('MINIBATCH_SIZE'))

HORIZONTAL_PAD = float(os.getenv('HORIZONTAL_PAD'))
