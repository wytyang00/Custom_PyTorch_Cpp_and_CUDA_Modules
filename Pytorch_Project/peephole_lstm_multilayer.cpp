#include <torch/torch.h>
#include <vector>

// CPU forward / backward definitions

std::vector<at::Tensor> peephole_lstm_multi_layer_cpu_forward(
	at::Tensor &input,
	at::Tensor &weight_ih,
	at::Tensor &weight_hh,
	at::Tensor &weight_ch,
	at::Tensor &bias,
	at::Tensor &old_h,
	at::Tensor &old_cell,
	double &dropout_p,
	bool &training)
{
	const auto sequence_length = input.size(0);
	const auto batch_size = input.size(1);
	const auto input_size = input.size(2);
	const auto state_size = old_h.size(1);

	at::Tensor outputs = at::empty(weight_ih.type(), { sequence_length, batch_size, state_size });
	at::Tensor tanh_new_cells = at::empty(old_cell.type(), { sequence_length, batch_size, state_size });
	at::Tensor gates = at::matmul(input, weight_ih.transpose(0, 1));
	at::Tensor X = at::empty(weight_ih.type(), { sequence_length, batch_size, (input_size + (2 * state_size)) });
	X.slice(2, 2 * state_size) = input;

	const auto weight_hc_h_t = at::cat({ weight_hh, weight_ch }, 1).transpose(0, 1);

	at::Tensor hc;
	at::Tensor gate_weights;
	std::vector<at::Tensor> sig_gates;
	at::Tensor tanh_gate;
	for (int i = 0; i < sequence_length; i++)
	{
		hc = at::cat({ old_h, old_cell }, /*dim=*/1);
		X[i].slice(1, 0, 2 * state_size) = hc;

		gate_weights = gates[i] + at::addmm(bias, hc, weight_hc_h_t);
		sig_gates = gate_weights.slice(1, 0, 3 * state_size).sigmoid().chunk(3, 1);
		tanh_gate = gate_weights.slice(1, 3 * state_size).tanh();

		old_cell = old_cell * sig_gates[0] + tanh_gate * sig_gates[1];
		auto tanh_cell = old_cell.tanh();
		tanh_new_cells[i] = tanh_cell;
		old_h = tanh_cell * sig_gates[2];

		gates[i] = at::cat({ sig_gates[0], sig_gates[1], sig_gates[2], tanh_gate }, 1);
		outputs[i] = old_h;
	}

	at::Tensor dropout;
	if (dropout_p <= 0. || !training) { dropout = at::ones_like(outputs); }
	else
	{
		if (dropout_p >= 1.) { dropout = at::zeros_like(outputs); outputs *= 0; }
		else { dropout = at::bernoulli(at::empty_like(outputs), (1 - dropout_p)).div(1 - dropout_p); outputs *= dropout; }
	}

	return { outputs,
		old_h,
		old_cell,
		tanh_new_cells,
		dropout,
		gates,
		X,
		at::cat({ weight_hh, weight_ch, weight_ih }, /*dim=*/1) };
}

std::vector<at::Tensor> peephole_lstm_cpu_backward(
	at::Tensor &grad_output,
	at::Tensor &grad_h,
	at::Tensor &grad_cell,
	at::Tensor &tanh_new_cells,
	at::Tensor &dropout,
	at::Tensor &gates,
	at::Tensor &X,
	at::Tensor &weights)
{
	const auto state_size = grad_h.size(1);
	const int input_size = X.size(2) - (2 * state_size);

	at::Tensor grad_inputs = at::empty(X.type(), { X.size(0), X.size(1), input_size });

	grad_output *= dropout;

	const auto forget_gates = gates.slice(2, 0, state_size);
	const auto output_gates = gates.slice(2, 2 * state_size, 3 * state_size);

	gates = at::cat({ X.slice(/*dim=*/2, state_size, 2 * state_size),
					  gates.slice(/*dim=*/2, 3 * state_size),
					  tanh_new_cells,
					  gates.slice(/*dim=*/2, state_size, 2 * state_size) }, /*dim=*/2)
		* at::cat({ (gates.slice(/*dim=*/2, 0, 3 * state_size) * (1 - gates.slice(/*dim=*/2, 0, 3 * state_size))),
					(1 - gates.slice(/*dim=*/2, 3 * state_size).pow(2)) }, /*dim=*/2);

	tanh_new_cells = (1 - tanh_new_cells.pow(2)) * output_gates;

	at::Tensor grad_new_cell;
	at::Tensor grad_X;
	for (int i = (X.size(0) - 1); i >= 0; i--)
	{
		grad_h += grad_output[i];

		grad_new_cell = tanh_new_cells[i] * grad_h + grad_cell;

		grad_cell = forget_gates[i] * grad_new_cell;

		gates[i] *= at::cat({ grad_new_cell, grad_new_cell, grad_h, grad_new_cell }, /*dim=*/1);

		grad_X = gates[i].mm(weights);
		grad_h = grad_X.slice(/*dim=*/1, 0, state_size);
		grad_cell += grad_X.slice(/*dim=*/1, state_size, 2 * state_size);
		grad_inputs[i] = grad_X.slice(/*dim=*/1, 2 * state_size);
	}
	auto d_weights = at::mm(gates.flatten(0, 1).t(), X.flatten(0, 1));
	auto d_bias = gates.sum({ 0, 1 }, false);

	return { grad_h, grad_cell, grad_inputs, d_weights.slice(1, 2 * state_size), d_weights.slice(1, 0, state_size), d_weights.slice(1, state_size, 2 * state_size), d_bias };
}

// CUDA forward / backward declarations
/* Not available yet
std::vector<at::Tensor> peephole_lstm_cuda_forward(
	at::Tensor const &input,
	at::Tensor const &weight_ih,
	at::Tensor const &weight_hh,
	at::Tensor const &weight_ch,
	at::Tensor const &bias,
	at::Tensor const &old_h,
	at::Tensor const &old_cell,
	double const &dropout_p,
	bool const &training);

std::vector<at::Tensor> peephole_lstm_cuda_backward(
	at::Tensor &grad_output,
	at::Tensor &grad_h,
	at::Tensor &grad_cell,
	at::Tensor &tanh_new_cells,
	at::Tensor const &dropout,
	at::Tensor &gates,
	at::Tensor const &X,
	at::Tensor const &weights);
*/

// C++ interface

#define CHECK_DIM(x, dimension) AT_ASSERTM(x.dim() == dimension, #x " must be a " #dimension "D tensor")
#define CHECK_DIM_MATCH(t1, d1, t2, d2) AT_ASSERTM(t1.size(d1) == t2.size(d2), "size of dimension " #d1 " in " #t1 " must match the size of dimension " #d2 " in " #t2)

#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_TENSOR(x) CHECK_CONTIGUOUS(x) // It used to contain device checking, but it was removed as I implemented CPU versions of functions alongside with the original functions.
											// Now, it's basically just contiguity check.
#define CHECK_INTERVAL(x, low, high) AT_ASSERTM((x >= low) && (x <= high), #x " must be within the interval ["#low", "#high"]")

#define CHECK_INPUT(x) CHECK_TENSOR(x);CHECK_DIM(x, 3)
#define CHECK_HIDDENS(x, h, c) CHECK_TENSOR(h);CHECK_TENSOR(c);CHECK_DIM(h, 2);CHECK_DIM(c, 2);CHECK_DIM_MATCH(h, 0, x, 1);CHECK_DIM_MATCH(c, 0, x, 1);CHECK_DIM_MATCH(h, 1, c, 1)
#define CHECK_WEIGHT(w, t, feature_dim) CHECK_TENSOR(w);CHECK_DIM(w, 2);CHECK_DIM_MATCH(w, 1, t, feature_dim)
#define CHECK_BIAS(b, h) CHECK_TENSOR(b);CHECK_DIM(b, 1);AT_ASSERTM(b.size(0) == 4 * h.size(1), "size of bias must be ", h.size(1), " but the given size was ", b.size(0))

std::vector<at::Tensor> peephole_lstm_forward(
	at::Tensor input,
	at::Tensor weight_ih,
	at::Tensor weight_hh,
	at::Tensor weight_ch,
	at::Tensor bias,
	at::Tensor old_h,
	at::Tensor old_cell,
	double dropout_p,
	bool training)
{
	CHECK_INPUT(input);
	CHECK_HIDDENS(input, old_h, old_cell);
	CHECK_WEIGHT(weight_ih, input, 2);
	CHECK_WEIGHT(weight_hh, old_h, 1);
	CHECK_WEIGHT(weight_ch, old_cell, 1);
	CHECK_BIAS(bias, old_h);
	CHECK_INTERVAL(dropout_p, 0, 1);

	bool use_cuda = input.is_cuda();

	AT_ASSERTM((use_cuda == weight_ih.is_cuda()) || (use_cuda == weight_hh.is_cuda()) || (use_cuda == weight_ch.is_cuda())
			   || (use_cuda == bias.is_cuda()) || (use_cuda == old_h.is_cuda()) || (use_cuda == old_cell.is_cuda()),
			   "### All tensors must be located in either CPU or CUDA devices together, but some of the given tensors are in a different device ###");

	if (false)
	{
		//return peephole_lstm_multi_layer_cuda_forward(input, weight_ih, weight_hh, weight_ch, bias, old_h, old_cell, dropout_p, training);
	}
	else
	{
		return peephole_lstm_multi_layer_cpu_forward(input, weight_ih, weight_hh, weight_ch, bias, old_h, old_cell, dropout_p, training);
	}
}

std::vector<at::Tensor> peephole_lstm_backward(
	at::Tensor grad_output,
	at::Tensor grad_h,
	at::Tensor grad_cell,
	at::Tensor tanh_new_cells,
	at::Tensor dropout,
	at::Tensor gates,
	at::Tensor X,
	at::Tensor weights)
{
	CHECK_TENSOR(grad_output);
	CHECK_TENSOR(grad_h);
	CHECK_TENSOR(grad_cell);
	CHECK_TENSOR(tanh_new_cells);
	CHECK_TENSOR(dropout);
	CHECK_TENSOR(gates);
	CHECK_TENSOR(X);
	CHECK_TENSOR(weights);

	bool use_cuda = grad_output.is_cuda();

	AT_ASSERTM((use_cuda == grad_h.is_cuda()) || (use_cuda == grad_cell.is_cuda()) || (use_cuda == tanh_new_cells.is_cuda()) || (use_cuda == dropout.is_cuda())
			   || (use_cuda == gates.is_cuda()) || (use_cuda == X.is_cuda()) || (use_cuda == weights.is_cuda()),
			   "### All tensors must be located in either CPU or CUDA devices together, but some of the given tensors are in a different device ###");

	if (use_cuda)
	{
		return peephole_lstm_cuda_backward(grad_output, grad_h, grad_cell, tanh_new_cells, dropout, gates, X, weights);
	}
	else
	{
		return peephole_lstm_cpu_backward(grad_output, grad_h, grad_cell, tanh_new_cells, dropout, gates, X, weights);
	}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("forward", &peephole_lstm_forward, "Peephole LSTM forward (CUDA)");
	m.def("backward", &peephole_lstm_backward, "Peephole LSTM backward (CUDA)");
}
