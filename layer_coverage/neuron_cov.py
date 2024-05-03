import sys

sys.path.append('../')

import numpy as np
from utils import get_layer_outs_new, percent_str, getimgs, getimgs2
from collections import defaultdict
from model import Model


def default_scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
            intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin

    return X_scaled



class NeuronCoverage:
    """
    Implements Neuron Coverage metric from "DeepXplore: Automated Whitebox Testing of Deep Learning Systems" by Pei
    et al.

    Supports incremental measurements using which one can observe the effect of new inputs to the coverage
    values.
    """

    def __init__(self, isimgnet, model, scaler=default_scale, threshold=0, skip_layers=None, batch=25,size=224):
        self.activation_table = defaultdict(bool)
        self.isimgnet = isimgnet
        self.model = model
        self.scaler = scaler
        self.threshold = threshold#0.7
        self.skip_layers = skip_layers = ([] if skip_layers is None else skip_layers)
        self.batch = batch
        self.size = size

    def get_measure_state(self):
        return [self.activation_table]

    def set_measure_state(self, state):
        self.activation_table = state[0]

    def test(self, test_inputs):
        """
        :param test_inputs: Inputs
        :return: Tuple containing the coverage and the measurements used to compute the coverage. 0th element is the
        percentage neuron coverage value.
        """
        print(self.threshold)
        print(type(self.model))
        # print(test_inputs.shape)
        print(len(test_inputs))
        # batch = 25
        input_batches = [test_inputs[i:i+self.batch] for i in range(0,len(test_inputs),self.batch)]
        print('batch num: ', len(input_batches))
        for b in input_batches:
            print('take a batch')
            imgs = getimgs(b, self.size)
            if  isinstance(self.model,Model):
                # outs = self.model.inference_single(test_inputs[:5])# 224  python run.py -M neural_networks/frozen_inference_graph -DS cifar10 -A nc
                outs = []
                # ins = []
                # for i in test_inputs:
                #     # i = np.resize(i,(224,224,3))
                #     ins.append(i)
                #     # outs.append(self.model.inference_single([i]))#[[],[],[],[],[]]
                outs, outputnames = self.model.inference_single(imgs)
                # print(np.array(ins).shape)
                # print(np.array(outs).shape)
            else:
                outs = get_layer_outs_new(self.model, imgs[:5], self.skip_layers)
            # print(len(outs))
            layer_all = []
            layer_covered = []
            t = []
            layer_covercnt = 0
            used_inps = []
            nc_cnt = 0
            # exit(-1)
            for layer_index, layer_out in enumerate(outs):  # layer_out is output of layer for all inputs[[[sample num],[],[]],[],[layer num]]
                # print(len(layer_out))
                inp_cnt = 0
                for out_for_input in layer_out:  # out_for_input is output of layer for single input
                    out_for_input = self.scaler(out_for_input)
                    # print(out_for_input.shape)
                    # print(len(out_for_input.flatten()))
                    # print(out_for_input)
                    for neuron_index in range(out_for_input.shape[-1]):
                        # print(out_for_input[..., neuron_index])
                        if not self.activation_table[(layer_index, neuron_index)] and np.mean(
                                out_for_input[..., neuron_index]) > self.threshold and inp_cnt not in used_inps:
                            used_inps.append(inp_cnt)
                            nc_cnt += 1
                        self.activation_table[(layer_index, neuron_index)] = self.activation_table[
                                                                                (layer_index, neuron_index)] or np.mean(
                            out_for_input[..., neuron_index]) > self.threshold
                    inp_cnt += 1

        for layer_index, layer_out in enumerate(outs):
            layer_all.append(layer_out[0].shape[-1])
            for neuron_index in range(layer_out[0].shape[-1]):
                if self.activation_table[(layer_index, neuron_index)]:
                    layer_covercnt += 1
            t.append(layer_covercnt)
            layer_covered.append(layer_covercnt/layer_all[layer_index])
            layer_covercnt = 0

        covered = len([1 for c in self.activation_table.values() if c])
        total = len(self.activation_table.keys())
        # print(self.activation_table)
        print('activated')
        print(t)
        print('total')
        print(layer_all)
        print('activated/total')
        print(layer_covered)
        print(covered)
        print(total)
        print('layer num:', len(layer_covered))
        for i in range(len(layer_covered)):
            print(i,' ', outputnames[i])
            if layer_covered[i]==0.0:
                print('0.0 ',i,' ', outputnames[i])
            if layer_covered[i]>0.9:
                print('>0.9',i,' ', outputnames[i])
            # if layer_covered[i]>0.1:
            #     print('>0.1',i,' ', outputnames[i])
        # for l, n in self.activation_table.keys():在这里写if self.activation_table[(layer_index, neuron_index)]:
        #                 layer_covercnt += 1

        return percent_str(covered, total), covered, total, outs, nc_cnt
