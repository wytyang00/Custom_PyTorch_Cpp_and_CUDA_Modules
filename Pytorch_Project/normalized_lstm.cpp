#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <vector>
#include <iostream>

inline void normalized_peephole_lstm_cell_forward(
	int const &sequence_index,
	int64_t const &state_size,
	at::Tensor const &input_h,
	at::Tensor const &weight_hc_h,
	at::Tensor const &bias,
	at::Tensor &hidden,
	at::Tensor &cell,
	at::Tensor &tanh_new_cells,
	at::Tensor &gates,
	at::Tensor &X,
	at::Tensor &outputs)
{
	auto hc = at::cat({ hidden, cell }, /*dim=*/1);
	X[sequence_index].slice(1, 0, 2 * state_size) = hc;

	auto gate_weights = input_h + at::addmm(bias, hc, weight_hc_h);
	auto sigmoid_gate_list = at::sigmoid(gate_weights.slice(/*dim=*/1, 0, 3 * state_size)).chunk(3, /*dim=*/1); //sig_forget, sig_input, sig_output
	auto tanh_gate = at::tanh(gate_weights.slice(/*dim=*/1, 3 * state_size)); //tanh_candidate_cell

	cell = cell * sigmoid_gate_list[0] + tanh_gate * sigmoid_gate_list[1];
	auto tanh_cell = at::tanh(cell);
	tanh_new_cells[sequence_index] = tanh_cell;
	hidden = tanh_cell * sigmoid_gate_list[2];

	gates[sequence_index] = at::cat({ sigmoid_gate_list[0], sigmoid_gate_list[1], sigmoid_gate_list[2], tanh_gate }, /*dim=*/1);
	outputs[sequence_index] = hidden;

	return;
}

inline void normalized_peephole_lstm_cell_backward(
	int const &sequence_index,
	int64_t const &state_size,
	at::Tensor const &grad_output,
	at::Tensor &d_h,
	at::Tensor &d_cell,
	at::Tensor const &dropout,
	at::Tensor const &d_tanh_of_new_cell,
	at::Tensor const &forget_gate,
	at::Tensor const &output_gate,
	at::Tensor const &d_gates_mult,
	at::Tensor const &X,
	at::Tensor const &weights,
	at::Tensor &d_inputs,
	at::Tensor &d_weights,
	at::Tensor &d_bias)
{
	d_h = d_h + (grad_output * dropout);

	auto d_tanh_new_cell = output_gate * d_h;
	auto d_new_cell = d_tanh_of_new_cell * d_tanh_new_cell + d_cell;

	d_cell = forget_gate * d_new_cell;

	auto d_gates = d_gates_mult * at::cat({ d_new_cell, d_h, d_new_cell, d_new_cell }, /*dim=*/1);

	d_weights += d_gates.t().mm(X);
	d_bias += d_gates.sum(/*dim=*/0, /*keepdim=*/false);
	auto d_X = d_gates.mm(weights);
	d_h = d_X.slice(/*dim=*/1, 0, state_size);
	d_cell += d_X.slice(/*dim=*/1, state_size, 2 * state_size);
	d_inputs[sequence_index] = d_X.slice(/*dim=*/1, 2 * state_size);

	return;
}

std::vector<at::Tensor> normalized_peephole_lstm_forward(
	at::Tensor const &input,
	at::Tensor const &weight_ih,
	at::Tensor const &weight_hh,
	at::Tensor const &weight_ch,
	at::Tensor const &bias,
	at::Tensor const &gammas,
	at::Tensor const &beta,
	double const &epsilon,
	bool const &training,
	at::Tensor const &running_means,
	at::Tensor const &running_stds,
	at::Tensor const &old_h,
	at::Tensor const &old_cell,
	double const &dropout_p)
{
	const double momentum = 0.1;
	const auto sequence_length = input.size(0);
	const auto batch_size = input.size(1);
	const auto input_size = input.size(2);
	const auto state_size = old_h.size(1);
	at::Tensor output = at::empty(weight_ih.type(), { sequence_length, batch_size, state_size });
	at::Tensor old_cells = at::empty(old_cell.type(), { sequence_length, batch_size, state_size });
	at::Tensor tanh_new_cells = at::empty(old_cell.type(), { sequence_length, batch_size, state_size });
	at::Tensor gates = at::empty(weight_ih.type(), { sequence_length, batch_size, 4 * state_size });
	at::Tensor X = at::empty(weight_ih.type(), { sequence_length, batch_size, (input_size + (2 * state_size)) });
	X.slice(2, 2 * state_size) = input;
	at::Tensor new_running_means_and_std = at::stack({ running_means, running_stds }, /*dim=*/0);

	const auto input_h = at::matmul(input, weight_ih.transpose(0, 1));

	if (training)
	{
		auto input_h_mean_and_std = at::stack({ input_h.mean(/*dim=*/1, /*keepdim=*/true), input_h.std(/*dim=*/1, /*unbiased=*/false, /*keepdim=*/true) }, /*dim=*/0);
		const auto input_h_norm = (input_h - input_h_mean_and_std[0]) / (input_h_mean_and_std[1] + epsilon) * gammas[0].view({ 1, 1, 4 * state_size });

		input_h_mean_and_std = input_h_mean_and_std.unsqueeze(2) * momentum;
		auto multiplier = at::full(input_h_mean_and_std.type(), { 1, input_h_mean_and_std.size(1) }, (1 - momentum));
		multiplier = multiplier.pow(at::arange(input_h_mean_and_std.type(), (input_h_mean_and_std.size(1) - 1), -1, -1).view(multiplier.sizes()));
		input_h_mean_and_std = input_h_mean_and_std.mul(multiplier).sum(/*dim=*/1, /*keepdim=*/false);
		new_running_means_and_std *= multiplier[0][0].mul(momentum);
		new_running_means_and_std.select(1, 0) += input_h_mean_and_std;
	}
	else
	{
		const auto input_h_norm = (input_h - new_running_means_and_std[0][0]) / (new_running_means_and_std[1][0] + epsilon) * gammas[0].view({ 1, 1, 4 * state_size });
	}

	const auto weight_hc_h_t = at::cat({ weight_hh, weight_ch }, 1).transpose(0, 1);
	auto hidden = old_h;
	auto cell = old_cell;

	for (int i = 0; i < input.size(0); i++)
	{
		peephole_lstm_cell_forward(i, state_size, input_h[i], weight_hc_h_t, bias, hidden, cell, tanh_new_cells, gates, X, output);
	}

	at::Tensor dropout;
	if (dropout_p <= 0. || !training) { dropout = at::ones_like(output); }
	else
	{
		if (dropout_p >= 1.) { dropout = at::zeros_like(output); output *= 0; }
		else { dropout = at::bernoulli(at::empty_like(output), (1 - dropout_p)).div(1 - dropout_p); output *= dropout; }
	}

	return { output,
		hidden,
		cell,
		tanh_new_cells,
		dropout,
		gates,
		X };
}

std::vector<at::Tensor> normalized_peephole_lstm_backward(
	at::Tensor const &grad_output,
	at::Tensor const &grad_h,
	at::Tensor const &grad_cell,
	at::Tensor const &new_cell,
	at::Tensor const &tanh_new_cells,
	at::Tensor const &dropout,
	at::Tensor const &gates,
	at::Tensor const &X,
	at::Tensor const &weight_ih,
	at::Tensor const &weight_hh,
	at::Tensor const &weight_ch)
{
	const auto weights = at::cat({ weight_ih, weight_hh, weight_ch }, 1);
	const auto input_size = weight_ih.size(1);
	const int64_t state_size = grad_h.size(1);
	at::Tensor d_input = at::empty(X.type(), { X.size(0), X.size(1), input_size });
	at::Tensor d_weights = at::zeros_like(weights);
	at::Tensor d_bias = at::zeros(weights.type(), { weights.size(0) });

	auto d_gates_mult = at::cat(
		{
			X.slice(/*dim=*/2, state_size, 2 * state_size),
			gates.slice(/*dim=*/2, 3 * state_size),
			tanh_new_cells,
			gates.slice(/*dim=*/2, state_size, 2 * state_size)
		}, /*dim=*/2);
	d_gates_mult *= at::cat({ (gates.slice(/*dim=*/2, 0, 3 * state_size) * (1 - gates.slice(/*dim=*/2, 0, 3 * state_size))),
							  (1 - gates.slice(/*dim=*/2, 3 * state_size).pow(2)) }, /*dim=*/2);
	const auto d_tanh_of_new_cells = (1 - tanh_new_cells.pow(2));

	const auto forget_gate = gates.slice(/*dim=*/2, 0, state_size);
	const auto output_gate = gates.slice(/*dim=*/2, 2 * state_size, 3 * state_size);

	at::Tensor d_h = grad_h;
	at::Tensor d_cell = grad_cell;

	for (int i = (X.size(0) - 1); i >= 0; i--)
	{
		peephole_lstm_cell_backward(i, state_size, grad_output[i], d_h, d_cell, dropout[i], d_tanh_of_new_cells[i],
									forget_gate[i], output_gate[i],
									d_gates_mult[i],
									X[i], weights, d_input, d_weights, d_bias);
	}

	return { d_h, d_cell, d_input, d_weights.slice(1, 0, input_size), d_weights.slice(1, input_size, input_size + state_size), d_weights.slice(1, input_size + state_size), d_bias };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("forward", &normalized_peephole_lstm_forward, "Normalized Peephole LSTM forward");
	m.def("backward", &normalized_peephole_lstm_backward, "Normalized Peephole LSTM backward");
}
