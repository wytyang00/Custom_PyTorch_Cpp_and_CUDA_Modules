#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <vector>

std::vector<at::Tensor> peephole_lstm_forward(
	at::Tensor input,
	at::Tensor weight_ih,
	at::Tensor weight_hh,
	at::Tensor weight_ch,
	at::Tensor bias,
	at::Tensor hidden,
	at::Tensor cell,
	double dropout_p,
	bool training)
{
	const auto sequence_length = input.size(0);
	const auto batch_size = input.size(1);
	const auto input_size = input.size(2);
	const auto state_size = hidden.size(1);

	// Outputs and variables containers
	at::Tensor outputs = at::empty(weight_ih.type(), { sequence_length, batch_size, state_size });
	at::Tensor tanh_new_cells = at::empty(cell.type(), { sequence_length, batch_size, state_size }); // tanh(cell) values are useful for backpropagation
	at::Tensor gates = at::matmul(input, weight_ih.transpose(0, 1)); // inputs are already provided, so they can be computed all at once at this time
	at::Tensor X = at::empty(weight_ih.type(), { sequence_length, batch_size, (input_size + (2 * state_size)) }); // inputs, hiddens, cells combined and stored
	X.slice(2, 2 * state_size) = input; // all inputs are already given

	const auto weight_hc_h_t = at::cat({ weight_hh,
									     at::cat({ weight_ch, at::zeros(weight_ch.type(), { state_size, state_size }) }, 0) }, 1).transpose(0, 1); // convenient and efficient form for matrix multiplication

	// Dropout for hidden and output
	at::Tensor dropout;
	if (dropout_p <= 0. || !training) { dropout = at::ones(outputs.type(), { 2, sequence_length, batch_size, state_size }); }
	else
	{
		if (dropout_p >= 1.) { dropout = at::zeros(outputs.type(), { 2, sequence_length, batch_size, state_size }); }
		else { dropout = at::bernoulli(at::empty(outputs.type(), { 2, sequence_length, batch_size, state_size }), (1 - dropout_p)).div(1 - dropout_p); }
	}

	// Temporary variables for use in the loop
	at::Tensor hc;
	at::Tensor gate_weights;
	std::vector<at::Tensor> sig_gates;
	at::Tensor tanh_gate;
	for (int i = 0; i < sequence_length; i++)
	{
		hidden *= dropout[0][i];
		hc = at::cat({ hidden, cell }, /*dim=*/1);
		X[i].slice(1, 0, 2 * state_size) = hc;

		gate_weights = gates[i] + at::addmm(bias, hc, weight_hc_h_t);
		sig_gates = gate_weights.slice(1, 0, 3 * state_size).sigmoid().chunk(3, 1);
		tanh_gate = gate_weights.slice(1, 3 * state_size).tanh();

		//cell = cell * sig_gates[0] + tanh_gate * sig_gates[1];
		cell = at::addcmul(tanh_gate * sig_gates[1], cell, sig_gates[0]);
		auto tanh_cell = cell.tanh();
		tanh_new_cells[i] = tanh_cell;
		hidden = tanh_cell * sig_gates[2];

		gates[i] = at::cat({ sig_gates[0], sig_gates[1], sig_gates[2], tanh_gate }, 1);
		outputs[i] = hidden;
	}

	outputs *= dropout[1];

	return { outputs,
		hidden,
		cell,
		tanh_new_cells,
		dropout,
		gates,
		X };
}

std::vector<at::Tensor> peephole_lstm_backward(
	at::Tensor grad_output,
	at::Tensor grad_h,
	at::Tensor grad_cell,
	at::Tensor tanh_new_cells,
	at::Tensor dropout,
	at::Tensor gates,
	at::Tensor X,
	at::Tensor weight_ih,
	at::Tensor weight_hh,
	at::Tensor weight_ch)
{
	const auto state_size = grad_h.size(1);
	const int input_size = X.size(2) - (2 * state_size);

	const auto weights = at::cat({ weight_hh, at::cat({ weight_ch, at::zeros(weight_ch.type(), { state_size, state_size }) }, 0), weight_ih }, 1);
	auto grad_inputs = at::empty(X.type(), { X.size(0), X.size(1), input_size });

	grad_output *= dropout[1];

	const auto forget_gates = gates.slice(2, 0, state_size);
	const auto output_gates = gates.slice(2, 2 * state_size, 3 * state_size);

	gates = at::mul(at::cat({ X.slice(/*dim=*/2, state_size, 2 * state_size),
					          gates.slice(/*dim=*/2, 3 * state_size),
					          tanh_new_cells,
					          gates.slice(/*dim=*/2, state_size, 2 * state_size) }, /*dim=*/2),
				    at::cat({ (gates.slice(/*dim=*/2, 0, 3 * state_size) * (1 - gates.slice(/*dim=*/2, 0, 3 * state_size))),
					          (1 - gates.slice(/*dim=*/2, 3 * state_size).pow(2)) }, /*dim=*/2));

	tanh_new_cells = at::mul(1 - tanh_new_cells.pow(2), output_gates);

	at::Tensor grad_new_cell;
	at::Tensor grad_X;
	for (int i = (X.size(0) - 1); i >= 0; i--)
	{
		grad_h += grad_output[i];

		grad_new_cell = at::addcmul(grad_cell, tanh_new_cells[i], grad_h);

		gates[i] *= at::cat({ grad_new_cell, grad_new_cell, grad_h, grad_new_cell }, /*dim=*/1);

		grad_X = at::mm(gates[i], weights);
		grad_h = at::mul(grad_X.slice(/*dim=*/1, 0, state_size), dropout[0][i]);
		grad_cell = at::addcmul(grad_X.slice(/*dim=*/1, state_size, 2 * state_size), forget_gates[i], grad_new_cell);
		grad_inputs[i] = grad_X.slice(/*dim=*/1, 2 * state_size);
	}
	auto d_weights = at::mm(gates.flatten(0, 1).t(), X.flatten(0, 1));
	auto d_bias = gates.sum({ 0, 1 }, false);

	return { grad_h,
		grad_cell,
		grad_inputs,
		d_weights.slice(1, 2 * state_size),
		d_weights.slice(1, 0, state_size),
		d_weights.slice(1, state_size, 2 * state_size).slice(0, 0, 3 * state_size),
		d_bias };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("forward", &peephole_lstm_forward, "Peephole LSTM forward");
	m.def("backward", &peephole_lstm_backward, "Peephole LSTM backward");
}
