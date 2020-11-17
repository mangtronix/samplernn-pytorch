from model import SampleRNN
import torch
from collections import OrderedDict
import os
import json
from trainer.plugins import GeneratorPlugin
import glob
import sys


'''Other comments: https://github.com/deepsound-project/samplernn-pytorch/issues/8'''

# Support some command line options
# Added by Mangtronix
# Michael Ang - https://michaelang.com

from optparse import OptionParser
parser = OptionParser()
parser.add_option('-d', '--dataset', help="Dataset name, e.g. 'lofi'")
parser.add_option('-l', '--length', help="Length of audio to generate in seconds", default=30)
parser.add_option('-c', '--checkpoint', help="Checkpoint name ('latest','best', or explicit name)", default='latest')
parser.add_option('-o', '--output', help="Output file name")
(options, args) = parser.parse_args()

if not options.dataset:
    parser.print_help()
    sys.exit(-1)

def find_results_path(dataset_name):
    paths = glob.glob('results/*' + dataset_name)
    if len(paths) < 1:
        print("No results found for " + dataset_name)
        raise("dataset not found")
    return paths[0] + '/'

def find_latest_checkpoint(results_path):
    files = glob.glob(results_path + "checkpoints/*")
    latest_file = max(files, key=os.path.getctime)
    return latest_file

def find_best_checkpoint(results_path):
    files = glob.glob(results_path + "checkpoints/best*")
    return files[-1]

def find_checkpoint(results_path, checkpoint_name):
    if checkpoint_name == 'best':
        return find_best_checkpoint(results_path)
    if checkpoint_name == 'latest':
        return find_latest_checkpoint(results_path)
    return results_path + "checkpoints/" + checkpoint_name
  

def get_checkpoint_name(checkpoint_path):
    return os.path.basename(os.path.normpath(checkpoint_path))

RESULTS_PATH=find_results_path(options.dataset) 
print("Using dataset at %s" % RESULTS_PATH)

PRETRAINED_PATH = find_checkpoint(RESULTS_PATH, options.checkpoint)
print("Using checkpoint %s" % PRETRAINED_PATH)

CHECKPOINT_NAME = get_checkpoint_name(PRETRAINED_PATH)
print("Checkpoint name is %s" % CHECKPOINT_NAME)

OUTPUT_NAME = options.dataset + "_" + CHECKPOINT_NAME

# Paths
#RESULTS_PATH = 'results/exp:TEST-frame_sizes:16,4-n_rnn:2-dataset:COGNIMUSE_eq_eq_pad/'
#RESULTS_PATH = 'results/exp:lofi-frame_sizes:16,4-n_rnn:2-dataset:lofi/'
#PRETRAINED_PATH = RESULTS_PATH + 'checkpoints/best-ep11-it2750'
#PRETRAINED_PATH = RESULTS_PATH + 'checkpoints/best-ep65-it79431'
# RESULTS_PATH = 'results/exp:TEST-frame_sizes:16,4-n_rnn:2-dataset:piano3/'
# PRETRAINED_PATH = RESULTS_PATH + 'checkpoints/best-ep21-it29610'
GENERATED_PATH = RESULTS_PATH + 'generated/'
if not os.path.exists(GENERATED_PATH):
    os.mkdir(GENERATED_PATH)

# Load model parameters from .json for audio generation
params_path = RESULTS_PATH + 'sample_rnn_params.json'
with open(params_path, 'r') as fp:
    params = json.load(fp)

# Create model with same parameters as used in training
model = SampleRNN(
    frame_sizes=params['frame_sizes'],
    n_rnn=params['n_rnn'],
    dim=params['dim'],
    learn_h0=params['learn_h0'],
    q_levels=params['q_levels'],
    weight_norm=params['weight_norm']
)
#model = model.cuda()

# Delete "model." from key names since loading the checkpoint automatically attaches it to the key names
pretrained_state = torch.load(PRETRAINED_PATH)
new_pretrained_state = OrderedDict()

for k, v in pretrained_state.items():
    layer_name = k.replace("model.", "")
    new_pretrained_state[layer_name] = v
    # print("k: {}, layer_name: {}, v: {}".format(k, layer_name, np.shape(v)))

# Load pretrained model
model.load_state_dict(new_pretrained_state)

# Generate Plugin
num_samples = 1  # params['n_samples']
sample_length = params['sample_length']
sample_rate = params['sample_rate']
sampling_temperature = params['sampling_temperature']

# Override from our options
sample_length = sample_rate * int(options.length)

print("Number samples: {}, sample_length: {}, sample_rate: {}".format(num_samples, sample_length, sample_rate))
print("Generating %d seconds of audio" % (sample_length / sample_rate))
generator = GeneratorPlugin(GENERATED_PATH, num_samples, sample_length, sample_rate, sampling_temperature)

# Call new register function to accept the trained model and the cuda setting
generator.register_generate(model.cuda(), params['cuda'])

# Generate new audio
# $$$ check if we already have generated audio and increment the file name
generator.epoch(OUTPUT_NAME)
GENERATED_FILEPATH = GENERATED_PATH + "ep" + OUTPUT_NAME + "-s1.wav"
print("Saved audio to %s " % GENERATED_FILEPATH)

if options.output:
    print("Moving to %s" % options.output)
    os.rename(GENERATED_FILEPATH, options.output)

