import numpy as np
import os
import sys
import json
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from models_torch import *
from utils import *
import tempfile
import argparse
import arithmeticcoding_fast
import struct
import time
import shutil

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

def loss_function(pred, target):
    loss = 1/np.log(2) * F.nll_loss(pred, target)
    return loss

def decompress(model, len_series, bs, vocab_size, timesteps, device, optimizer, scheduler, final_step=False):
    
    if not final_step:
        num_iters = len_series // bs
        print(num_iters)
        series_2d = np.zeros((bs,num_iters), dtype = np.uint8).astype('int')
        ind = np.array(range(bs))*num_iters

        f = [open(FLAGS.temp_file_prefix+'.'+str(i),'rb') for i in range(bs)]
        bitin = [arithmeticcoding_fast.BitInputStream(f[i]) for i in range(bs)]
        dec = [arithmeticcoding_fast.ArithmeticDecoder(32, bitin[i]) for i in range(bs)]

        prob = np.ones(vocab_size)/vocab_size
        cumul = np.zeros(vocab_size+1, dtype = np.uint64)
        cumul[1:] = np.cumsum(prob*10000000 + 1)

        # Decode first K symbols in each stream with uniform probabilities
        for i in range(bs):
            for j in range(min(timesteps, num_iters)):
                series_2d[i,j] = dec[i].read(cumul, vocab_size)

        cumul = np.zeros((bs, vocab_size+1), dtype = np.uint64)

        block_len = 20
        test_loss = 0
        batch_loss = 0
        start_time = time.time()
        for j in (range(num_iters - timesteps)):
            # Create Batch
            bx = Variable(torch.from_numpy(series_2d[:,j:j+timesteps])).to(device)
            
            with torch.no_grad():
                model.eval()
                pred, _ = model(bx)
                prob = torch.exp(pred).detach().cpu().numpy()
            cumul[:,1:] = np.cumsum(prob*10000000 + 1, axis = 1)

            # Decode with Arithmetic Encoder
            for i in range(bs):
                series_2d[i,j+timesteps] = dec[i].read(cumul[i,:], vocab_size)
            
            by = Variable(torch.from_numpy(series_2d[:, j+timesteps])).to(device)
            loss = loss_function(pred, by)
            test_loss += loss.item()
            batch_loss += loss.item()

            if (j+1) % 100 == 0:
                print("Iter {} Loss {:.4f} Moving Loss {:.4f}".format(j+1, test_loss/(j+1), batch_loss/100), flush=True)
                print("{} secs".format(time.time() - start_time))
                batch_loss = 0
                start_time = time.time()

            # Update Parameters of Combined Model
            if (j+1) % block_len == 0:
                model.train()
                optimizer.zero_grad()
                data_x = np.concatenate([series_2d[:, j + np.arange(timesteps) - p] for p in range(block_len)], axis=0)
                data_y = np.concatenate([series_2d[:, j + timesteps - p] for p in range(block_len)], axis=0)

                bx = Variable(torch.from_numpy(data_x)).to(device)
                by = Variable(torch.from_numpy(data_y)).to(device)
                pred1, pred2 = model(bx)
                loss2 = loss_function(pred2, by)
                loss = loss_function(pred1, by) + loss2
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()


        # close files
        for i in range(bs):
            bitin[i].close()
            f[i].close()
        return series_2d.reshape(-1)
    
    else:
        series = np.zeros(len_series, dtype = np.uint8).astype('int')
        f = open(FLAGS.temp_file_prefix+'.last','rb')
        bitin = arithmeticcoding_fast.BitInputStream(f)
        dec = arithmeticcoding_fast.ArithmeticDecoder(32, bitin)
        prob = np.ones(vocab_size)/vocab_size
        cumul = np.zeros(vocab_size+1, dtype = np.uint64)
        cumul[1:] = np.cumsum(prob*10000000 + 1)        

        for j in range(min(timesteps,len_series)):
            series[j] = dec.read(cumul, vocab_size)
        for i in range(len_series-timesteps):
            bx = Variable(torch.from_numpy(series[i:i+timesteps].reshape(1,-1))).to(device)
            with torch.no_grad():
                model.eval()
                pred, _ = model(bx)
                prob = torch.exp(pred).detach().cpu().numpy()
            cumul[1:] = np.cumsum(prob*10000000 + 1)
            series[i+timesteps] = dec.read(cumul, vocab_size)
        bitin.close()
        f.close()
        return series

def get_argument_parser():
    parser = argparse.ArgumentParser();
    parser.add_argument('--file_name', type=str, default='xor10_small_com',
                        help='The name of the input file')
    parser.add_argument('--output', type=str, default='xor10_small_decomp',
                        help='The name of the output file')
    parser.add_argument('--model_weights_path', type=str, default='xor10_small_bstrap',
                        help='Path to model weights')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU to use')
    return parser


def var_int_decode(f):
    byte_str_len = 0
    shift = 1
    while True:
        this_byte = struct.unpack('B', f.read(1))[0]
        byte_str_len += (this_byte & 127) * shift
        if this_byte & 128 == 0:
                break
        shift <<= 7
        byte_str_len += shift
    return byte_str_len

def main():
    os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu
    use_cuda = True

    FLAGS.temp_dir = 'temp'
    if os.path.exists(FLAGS.temp_dir):
        shutil.rmtree('temp')
    FLAGS.temp_file_prefix = FLAGS.temp_dir + "/compressed"
    if not os.path.exists(FLAGS.temp_dir):
        os.makedirs(FLAGS.temp_dir)

    f = open(FLAGS.file_name+'.params','r')
    params = json.loads(f.read())
    f.close()

    batch_size = params['bs']
    timesteps = params['timesteps']
    len_series = params['len_series']
    id2char_dict = params['id2char_dict']
    vocab_size = len(id2char_dict)

    # Break into multiple streams
    f = open(FLAGS.file_name+'.combined','rb')
    for i in range(batch_size):
        f_out = open(FLAGS.temp_file_prefix+'.'+str(i),'wb')
        byte_str_len = var_int_decode(f)
        byte_str = f.read(byte_str_len)
        f_out.write(byte_str)
        f_out.close()
    f_out = open(FLAGS.temp_file_prefix+'.last','wb')
    byte_str_len = var_int_decode(f)
    byte_str = f.read(byte_str_len)
    f_out.write(byte_str)
    f_out.close()
    f.close()

    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using device {}".format(device))

    series = np.zeros(len_series,dtype=np.uint8)

    bsdic = {'vocab_size': vocab_size, 'emb_size': 8,
        'length': timesteps, 'jump': 16,
        'hdim1': 8, 'hdim2': 16, 'n_layers': 2,
        'bidirectional': True}
    comdic = {'vocab_size': vocab_size, 'emb_size': 32,
        'length': timesteps, 'hdim': 8}


    # Select Model Parameters based on Alphabet Size
    if vocab_size >= 1 and vocab_size <=3:
        bsdic['hdim1'] = 8
        bsdic['hdim2'] = 16
        comdic['emb_size'] = 16
        comdic['hdim'] = 1024
      
    if vocab_size >= 4 and vocab_size <=9:
        bsdic['hdim1'] = 32
        bsdic['hdim2'] = 16
        comdic['emb_size'] = 16
        comdic['hdim'] = 2048

    if vocab_size >= 10 and vocab_size < 128:
        bsdic['hdim1'] = 128
        bsdic['hdim2'] = 128
        bsdic['emb_size'] = 16
        comdic['emb_size'] = 32
        comdic['hdim'] = 2048

    if vocab_size >= 128:
        bsdic['hdim1'] = 128
        bsdic['hdim2'] = 256
        bsdic['emb_size'] = 16
        comdic['emb_size'] = 32
        comdic['hdim'] = 2048


    # Define Model and load bootstrap weights
    bsmodel = BootstrapNN(**bsdic).to(device)
    bsmodel.load_state_dict(torch.load(FLAGS.model_weights_path))
    comdic['bsNN'] = bsmodel
    commodel = CombinedNN(**comdic).to(device)
    
    # Freeze Bootstrap Weights
    for name, p in commodel.named_parameters():
        if "bs" in name:
            p.requires_grad = False
    
    # Optimizer
    optimizer = optim.Adam(commodel.parameters(), lr=5e-4, betas=(0.0, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, threshold=1e-2, patience=1000, cooldown=10000, min_lr=1e-4, verbose=True)
    l = int(len(series)/batch_size)*batch_size
    
    series[:l] = decompress(commodel, l, batch_size, vocab_size, timesteps, device, optimizer, scheduler)
    if l < len_series - timesteps:
        series[l:] = decompress(commodel, len_series-l, 1, vocab_size, timesteps, device, optimizer, scheduler, final_step = True)
    else:
        f = open(FLAGS.temp_file_prefix+'.last','rb')
        bitin = arithmeticcoding_fast.BitInputStream(f)
        dec = arithmeticcoding_fast.ArithmeticDecoder(32, bitin) 
        prob = np.ones(vocab_size)/vocab_size
        
        cumul = np.zeros(vocab_size+1, dtype = np.uint64)
        cumul[1:] = np.cumsum(prob*10000000 + 1)        
        for j in range(l, len_series):
            series[j] = dec.read(cumul, vocab_size)
        
        bitin.close() 
        f.close()
    
    # Write to output
    f = open(FLAGS.output,'wb')
    f.write(bytearray([id2char_dict[str(s)] for s in series]))
    f.close()

    shutil.rmtree('temp')
    print("Done")
    


if __name__ == "__main__":
    parser = get_argument_parser()
    FLAGS = parser.parse_args()
    main()




