#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <vector>

inline void peephole_lstm_cell_forward(
	int const &sequence_index,
	int64_t const &state_size,
	at::Tensor const &weight_hc_h,
	at::Tensor const &bias,
	at::Tensor const &gate,
	at::Tensor &hidden,
	at::Tensor &cell,
	at::Tensor &tanh_new_cells,
	at::Tensor &gates,
	at::Tensor &X,
	at::Tensor &outputs)
{
	auto hc = at::cat({ hidden, cell }, /*dim=*/1);
	X[sequence_index].slice(1, 0, 2 * state_size) = hc;

	auto gate_weights = gate + at::addmm(bias, hc, weight_hc_h);
	auto sig_gates = gate_weights.slice(1, 0, 3 * state_size).sigmoid().chunk(3, 1);
	auto tanh_gate = gate_weights.slice(1, 3 * state_size).tanh();

	cell = cell * sig_gates[0] + tanh_gate * sig_gates[1];
	auto tanh_cell = cell.tanh();
	tanh_new_cells[sequence_index] = tanh_cell;
	hidden = tanh_cell * sig_gates[2];

	gates[sequence_index] = at::cat({ sig_gates[0], sig_gates[1], sig_gates[2], tanh_gate }, 1);
	outputs[sequence_index] = hidden;

	return;
}

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
	const auto sequence_length = input.size(0);
	const auto batch_size = input.size(1);
	const auto input_size = input.size(2);
	const auto state_size = old_h.size(1);

	at::Tensor output = at::empty(weight_ih.type(), { sequence_length, batch_size, state_size });
	at::Tensor tanh_new_cells = at::empty(old_cell.type(), { sequence_length, batch_size, state_size });
	at::Tensor gates = at::matmul(input, weight_ih.transpose(0, 1));
	at::Tensor X = at::empty(weight_ih.type(), { sequence_length, batch_size, (input_size + (2 * state_size)) });
	X.slice(2, 2 * state_size) = input;

	const auto weight_hc_h_t = at::cat({ weight_hh, weight_ch }, 1).transpose(0, 1);
	auto hidden = old_h;
	auto cell = old_cell;

	for (int i = 0; i < sequence_length; i++)
	{
		peephole_lstm_cell_forward(i, state_size, weight_hc_h_t, bias, gates[i], hidden, cell, tanh_new_cells, gates, X, output);
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
		X,
		at::cat({ weight_hh, weight_ch, weight_ih }, /*dim=*/1) };
}

inline void peephole_lstm_cell_backward(
	int const &sequence_index,
	int64_t const &state_size,
	at::Tensor const &grad_output,
	at::Tensor &d_h,
	at::Tensor &d_cell,
	at::Tensor const &d_tanh_of_new_cell,
	at::Tensor const &forget_gate,
	at::Tensor const &gates,
	at::Tensor const &weights,
	at::Tensor &d_inputs)
{
	d_h += grad_output;

	auto d_new_cell = d_tanh_of_new_cell * d_h + d_cell;

	d_cell = forget_gate * d_new_cell;

	gates[sequence_index] *= at::cat({ d_new_cell, d_new_cell, d_h, d_new_cell }, /*dim=*/1);

	auto d_X = gates[sequence_index].mm(weights);
	d_h = d_X.slice(/*dim=*/1, 0, state_size);
	d_cell += d_X.slice(/*dim=*/1, state_size, 2 * state_size);
	d_inputs[sequence_index] = d_X.slice(/*dim=*/1, 2 * state_size);

	return;
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
	const auto state_size = grad_h.size(1);
	const int input_size = X.size(2) - ( 2 * state_size );

	at::Tensor d_input = at::empty(X.type(), { X.size(0), X.size(1), input_size });
	//at::Tensor d_weights = at::zeros_like(weights);
	//at::Tensor d_bias = at::zeros(weights.type(), { weights.size(0) });

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

	//const int compute_length = 50;
	for (int i = (X.size(0) - 1); i >= 0; i--)
	{
		peephole_lstm_cell_backward(i, state_size, grad_output[i], grad_h, grad_cell, tanh_new_cells[i],
									forget_gates[i], gates, weights, d_input);
			
		//if (i % compute_length == 0)
		//{
		//	d_weights += at::matmul(gates.slice(0, i, i + compute_length).transpose(1, 2), X.slice(0, i, i + compute_length)).sum(/*dim=*/0, /*keepdim=*/false);
		//	d_bias += gates.slice(0, i, i + compute_length).sum(/*dims=*/{ 0, 1 }, /*keepdim=*/false);
		//}
	}
	auto d_weights = at::mm(gates.flatten(0, 1).t(), X.flatten(0, 1));
	auto d_bias = gates.sum({ 0, 1 }, false);

	return { grad_h, grad_cell, d_input, d_weights.slice(1, 2 * state_size), d_weights.slice(1, 0, state_size), d_weights.slice(1, state_size, 2 * state_size), d_bias };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("forward", &peephole_lstm_forward, "Peephole LSTM forward");
	m.def("backward", &peephole_lstm_backward, "Peephole LSTM backward");
}
