import os
import math

import torch
import torch.nn as nn
import torch.utils.cpp_extension as cpp_extension

_ln_peephole_lstm_layer = cpp_extension.load('ln_peephole_lstm_layer_cuda',
                                             [os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ln_peephole_lstm_layer_cuda.cpp'),
                                              os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ln_peephole_lstm_layer_cuda_kernel.cu')])

class LNPeepholeLSTMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight_ih, weight_hh, weight_ch, bias,
                gamma_f, gamma_i, gamma_g, gamma_o, gamma_cell, beta_cell,
                hidden, cell,
                epsilon, dropout_p,
                dropout_output, training):

        outputs = _ln_peephole_lstm_layer.forward(input, weight_ih, weight_hh, weight_ch, bias,
                                                      gamma_f, gamma_i, gamma_g, gamma_o, gamma_cell, beta_cell,
                                                      hidden, cell,
                                                      epsilon, dropout_p,
                                                      dropout_output, training)

        out, new_h, new_cell = outputs[:3]

        variables = outputs[3:] + [weight_ih, weight_hh, weight_ch,
                                   gamma_f, gamma_i, gamma_g, gamma_o, gamma_cell]
        ctx.save_for_backward(*variables)

        return out, new_h, new_cell

    @staticmethod
    def backward(ctx, grad_output, grad_h, grad_cell):
        outputs = _ln_peephole_lstm_layer.backward(
            grad_output.contiguous(), grad_h.contiguous(), grad_cell.contiguous(), *ctx.saved_tensors)

        (d_input, d_weight_ih, d_weight_hh, d_weight_ch, d_bias,
         d_gamma_f, d_gamma_i, d_gamma_o, d_gamma_g, d_gamma_cell, d_beta_cell,
         d_hidden, d_cell) = outputs

        return (d_input, d_weight_ih, d_weight_hh, d_weight_ch, d_bias,
                d_gamma_f, d_gamma_i, d_gamma_o, d_gamma_g, d_gamma_cell, d_beta_cell,
                d_hidden, d_cell,
                None, None,
                None, None)

class LNPeepholeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=False, dropout=0., dropout_on_output=True, eps=1e-05):
        if not 0 <= dropout <= 1:
            raise ValueError(f"Invalid dropout value : {dropout} dropout must be in range [0, 1].")

        super(LNPeepholeLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = bool(batch_first)
        self.dropout = float(dropout)
        self.dropout_on_output = bool(dropout_on_output)
        self.eps = eps

        self.register_parameter('weight_ih', nn.Parameter(torch.empty(4 * hidden_size, input_size)))
        self.register_parameter('weight_hh', nn.Parameter(torch.empty(4 * hidden_size, hidden_size)))
        self.register_parameter('weight_ch', nn.Parameter(torch.empty(3 * hidden_size)))
        self.register_parameter('bias', nn.Parameter(torch.empty(4 * hidden_size)))

        self.register_parameter('gamma_f', nn.Parameter(torch.empty(hidden_size)))
        self.register_parameter('gamma_i', nn.Parameter(torch.empty(hidden_size)))
        self.register_parameter('gamma_g', nn.Parameter(torch.empty(hidden_size)))
        self.register_parameter('gamma_o', nn.Parameter(torch.empty(hidden_size)))
        self.register_parameter('gamma_cell', nn.Parameter(torch.empty(hidden_size)))
        self.register_parameter('beta_cell', nn.Parameter(torch.empty(hidden_size)))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.input_size + 2 * self.hidden_size)
        self.weight_ih.data.uniform_(-stdv, +stdv)
        self.weight_hh.data.uniform_(-stdv, +stdv)
        self.weight_ch.data.uniform_(-stdv, +stdv)

        self.bias.data.zero_()
        self.bias.data[:self.hidden_size].fill_(1.)

        self.gamma_f.data.uniform_()
        self.gamma_i.data.uniform_()
        self.gamma_g.data.uniform_()
        self.gamma_o.data.uniform_()
        self.gamma_cell.data.uniform_()
        self.beta_cell.data.zero_()

    def forward(self, input, state):
        if self.batch_first:
            input = input.transpose(0, 1).contiguous()

        output, new_h, new_cell = LNPeepholeLSTMFunction.apply(
            input, self.weight_ih, self.weight_hh, self.weight_ch, self.bias,
            self.gamma_f, self.gamma_i, self.gamma_g, self.gamma_o, self.gamma_cell, self.beta_cell,
            state[0], state[1],
            self.eps, self.dropout, self.dropout_on_output, self.training)
        
        if self.batch_first:
            output = output.transpose(0, 1).contiguous()

        return output, (new_h, new_cell)

    def __repr__(self):
        return f"LNPeepholeLSTM(input_size={self.input_size}, hidden_size={self.hidden_size}, batch_first={self.batch_first}, dropout={self.dropout}, dropout_on_output={self.dropout_on_output}, eps={self.eps})"
