import numpy as np
import os
import sys
import json
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from utils import *
import tempfile
import argparse
import arithmeticcoding_fast
import struct
import shutil
from models_torch import *

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"



def loss_function(pred, target):
    loss = 1/np.log(2) * F.nll_loss(pred, target)
    return loss


def compress(model, X, Y, bs, vocab_size, timesteps, device, final_step=False):
    
    if not final_step:
        num_iters = (len(X)+timesteps) // bs
        ind = np.array(range(bs))*num_iters

        f = [open(FLAGS.temp_file_prefix+'.'+str(i),'wb') for i in range(bs)]
        bitout = [arithmeticcoding_fast.BitOutputStream(f[i]) for i in range(bs)]
        enc = [arithmeticcoding_fast.ArithmeticEncoder(32, bitout[i]) for i in range(bs)]

        prob = np.ones(vocab_size)/vocab_size
        cumul = np.zeros(vocab_size+1, dtype = np.uint64)
        cumul[1:] = np.cumsum(prob*10000000 + 1)


        # Encode first K symbols in each stream with uniform probabilities
        for i in range(bs):
            for j in range(min(timesteps, num_iters)):
                enc[i].write(cumul, X[ind[i],j])

        cumul = np.zeros((bs, vocab_size+1), dtype = np.uint64)

        for j in (range(num_iters - timesteps)):
            # Create Batches
            bx = Variable(torch.from_numpy(X[ind,:])).to(device)
            by = Variable(torch.from_numpy(Y[ind])).to(device)
            with torch.no_grad():
                model.eval()
                prob = torch.exp(model(bx)).detach().cpu().numpy()
            cumul[:,1:] = np.cumsum(prob*10000000 + 1, axis = 1)

            # Encode with Arithmetic Encoder
            for i in range(bs):
                enc[i].write(cumul[i,:], Y[ind[i]])
            ind = ind + 1

            if (j+1)%100 == 0:
                print("Step {}/{} ".format(j+1, num_iters - timesteps), flush=True)

        # close files
        for i in range(bs):
            enc[i].finish()
            bitout[i].close()
            f[i].close()
    
    else:
        f = open(FLAGS.temp_file_prefix+'.last','wb')
        bitout = arithmeticcoding_fast.BitOutputStream(f)
        enc = arithmeticcoding_fast.ArithmeticEncoder(32, bitout)
        prob = np.ones(vocab_size)/vocab_size
        cumul = np.zeros(vocab_size+1, dtype = np.uint64)
        cumul[1:] = np.cumsum(prob*10000000 + 1)        

        for j in range(timesteps):
            enc.write(cumul, X[0,j])
        for i in (range(len(X))):
            bx = Variable(torch.from_numpy(X[i:i+1,:])).to(device)
            with torch.no_grad():
                model.eval()
                prob = torch.exp(model(bx)).detach().cpu().numpy()
            cumul[1:] = np.cumsum(prob*10000000 + 1)
            enc.write(cumul, Y[i])
        enc.finish()
        bitout.close()
        f.close()
    
    return


def get_argument_parser():
    parser = argparse.ArgumentParser();
    parser.add_argument('--file_name', type=str, default='xor10_small',
                        help='The name of the input file')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU to use')
    parser.add_argument('--output', type=str, default='comp',
                        help='Name of the output file')
    parser.add_argument('--model_weights_path', type=str, default='file_bstrap',
                        help='Path to model parameters')
    parser.add_argument('--timesteps', type=int, default='64',
                        help='Number of timesteps')
    return parser


def var_int_encode(byte_str_len, f):
    while True:
        this_byte = byte_str_len&127
        byte_str_len >>= 7
        if byte_str_len == 0:
                f.write(struct.pack('B',this_byte))
                break
        f.write(struct.pack('B',this_byte|128))
        byte_str_len -= 1

def main():
    os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu

    batch_size=512
    timesteps=FLAGS.timesteps
    use_cuda = True

    with open("params_"+ FLAGS.file_name, 'r') as f:
        params = json.load(f)

    FLAGS.temp_dir = 'temp'
    if os.path.exists(FLAGS.temp_dir):
        os.system("rm -r {}".format(FLAGS.temp_dir))
    FLAGS.temp_file_prefix = FLAGS.temp_dir + "/compressed"
    if not os.path.exists(FLAGS.temp_dir):
        os.makedirs(FLAGS.temp_dir)

    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    sequence = np.load(FLAGS.file_name + ".npy")
    vocab_size = len(np.unique(sequence))
    sequence = sequence

    sequence = sequence.reshape(-1)
    series = sequence.copy()

    # Convert data into context and target symbols
    data = strided_app(series, timesteps+1, 1)
    X = data[:, :-1]
    Y = data[:, -1]

    params['len_series'] = len(series)
    params['bs'] = batch_size
    params['timesteps'] = timesteps

    with open(FLAGS.output+'.params','w') as f:
        json.dump(params, f, indent=4)

    bsdic = {'vocab_size': vocab_size, 'emb_size': 8,
        'length': timesteps, 'jump': 16,
        'hdim1': 8, 'hdim2': 16, 'n_layers': 2,
        'bidirectional': True}

    # Select Model Parameters based on Alphabet Size
    if vocab_size >= 1 and vocab_size <=3:
        bsdic['hdim1'] = 8
        bsdic['hdim2'] = 16
      
    if vocab_size >= 4 and vocab_size <=9:
        bsdic['hdim1'] = 32
        bsdic['hdim2'] = 16

    if vocab_size >= 10 and vocab_size < 128:
        bsdic['hdim1'] = 128
        bsdic['hdim2'] = 128
        bsdic['emb_size'] = 16

    if vocab_size >= 128:
        bsdic['hdim1'] = 128
        bsdic['hdim2'] = 256
        bsdic['emb_size'] = 16

    model = BootstrapNN(**bsdic).to(device)
    model.load_state_dict(torch.load(FLAGS.model_weights_path))

    l = int(len(series)/batch_size)*batch_size
    
    compress(model, X, Y, batch_size, vocab_size, timesteps, device)
    if l < len(series)-timesteps:
        compress(model, X[l:], Y[l:], 1, vocab_size, timesteps, device, final_step = True)
    else:
        f = open(FLAGS.temp_file_prefix+'.last','wb')
        bitout = arithmeticcoding_fast.BitOutputStream(f)
        enc = arithmeticcoding_fast.ArithmeticEncoder(32, bitout) 
        prob = np.ones(vocab_size)/vocab_size
        
        cumul = np.zeros(vocab_size+1, dtype = np.uint64)
        cumul[1:] = np.cumsum(prob*10000000 + 1)        
        for j in range(l, len(series)):
                enc.write(cumul, series[j])
        enc.finish()
        bitout.close() 
        f.close()
    
    print("Done")
    
    # combine files into one file
    f = open(FLAGS.output+'.combined','wb')
    for i in range(batch_size):
        f_in = open(FLAGS.temp_file_prefix+'.'+str(i),'rb')
        byte_str = f_in.read()
        byte_str_len = len(byte_str)
        var_int_encode(byte_str_len, f)
        f.write(byte_str)
        f_in.close()
    f_in = open(FLAGS.temp_file_prefix+'.last','rb')
    byte_str = f_in.read()
    byte_str_len = len(byte_str)
    var_int_encode(byte_str_len, f)
    f.write(byte_str)
    f_in.close()
    f.close()
    shutil.rmtree('temp')


if __name__ == "__main__":
    parser = get_argument_parser()
    FLAGS = parser.parse_args()
    main()




