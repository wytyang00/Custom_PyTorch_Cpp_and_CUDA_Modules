#include <torch/torch.h>
#include <vector>

std::vector<at::Tensor> ln_peephole_lstm_layer_forward(
	at::Tensor input,
	at::Tensor weight_ih,
	at::Tensor weight_hh,
	at::Tensor weight_ch,
	at::Tensor bias,
	at::Tensor gamma_ih,
	at::Tensor gamma_hh,
	at::Tensor gamma_ch,
	at::Tensor gamma_tanh_cell,
	at::Tensor beta_tanh_cell,
	at::Tensor hidden,
	at::Tensor cell,
	double epsilon,
	double dropout_p,
	bool training)
{
	AT_ASSERTM(input.dim() == 3, "### input must be a 3D tensor ###");
	const auto sequence_length = input.size(0);
	const auto batch_size = input.size(1);
	AT_ASSERTM((batch_size > 1) || (!training), "### batch size of 1 during training is invaild (for batch normalization, the batch size must be at least 2) ###");
	const auto input_size = input.size(2);
	const auto state_size = weight_ih.size(0) / 4;
	const auto state_size_3 = 3 * state_size;
	const auto gate_size = state_size_3 + state_size;

	at::Tensor output = at::zeros(weight_ih.type(), { sequence_length, batch_size, state_size });

	at::Tensor tanh_new_cells = at::zeros(cell.type(), { sequence_length, batch_size, state_size });
	at::Tensor bnorm_tanh_cells = at::zeros_like(tanh_new_cells);

	at::Tensor norm_collection = at::zeros(input.type(), { 3, sequence_length, batch_size, 4 * state_size });
	at::Tensor norm_gates_ih = norm_collection[0];
	at::Tensor norm_gates_hh = norm_collection[1];
	at::Tensor norm_gates_ch = norm_collection[2].slice(2, 0, state_size_3);
	at::Tensor norm_tanh_cells = norm_collection[2].slice(2, state_size_3);

	at::Tensor gates = at::matmul(input, weight_ih.transpose(0, 1));
	at::Tensor X = at::zeros(weight_ih.type(), { sequence_length, batch_size, (input_size + (2 * state_size)) });
	X.slice(2, 2 * state_size) = input;

	at::Tensor dropout;
	if (dropout_p <= 0. || !training) { dropout = at::ones(output.type(), { 2, sequence_length, batch_size, state_size }); }
	else
	{
		if (dropout_p >= 1.) { dropout = at::zeros(output.type(), { 2, sequence_length, batch_size, state_size }); }
		else { dropout = at::bernoulli(at::zeros(output.type(), { 2, sequence_length, batch_size, state_size }), (1 - dropout_p)).div(1 - dropout_p); }
	}
	auto dropout_hidden = dropout[0];
	auto dropout_output = dropout[1];

	at::Tensor stds_collection;

	auto hc = at::stack({ hidden, cell });

	const auto weight_hc_t = at::stack({ weight_hh, at::cat({ weight_ch, at::zeros(weight_ch.type(), { state_size, state_size }) }, 0) }).transpose(1, 2);
	const auto gammas_hh_ch = at::stack({ gamma_hh, at::cat({ gamma_ch, at::zeros(gamma_ch.type(), { state_size }) }, 0) }, 0).unsqueeze(1);
	at::Tensor ch_gate_pair;
	at::Tensor mean;
	at::Tensor std;
	at::Tensor current_gate;
	at::Tensor norm_gate;
	at::Tensor tanh_cell;
	at::Tensor norm_tanh_cell;

	stds_collection = at::zeros(gates.type(), { 4, sequence_length, batch_size });

	gates -= gates.mean(/*dim=*/2, /*keepdim=*/true);
	std = gates.var(/*dim=*/2, /*unbiased=*/false, /*keepdim=*/false).add(epsilon).sqrt();
	stds_collection[0] = std;
	gates /= std.unsqueeze(2);
	norm_collection[0] = gates;
	gates = at::addcmul(bias, gates, gamma_ih);

	// Forward Loop
	for (int i = 0; i < sequence_length; i++)
	{
		current_gate = gates[i];

		hc[0] *= dropout_hidden[i];
		X[i].slice(1, 0, state_size) = hc[0];
		X[i].slice(1, state_size, 2 * state_size) = hc[1];

		ch_gate_pair = at::matmul(hc, weight_hc_t);

		mean = at::stack({ ch_gate_pair[0].mean(/*dim=*/1, /*keepdim=*/true),
						   ch_gate_pair[1].slice(1, 0, state_size_3).mean(/*dim=*/1, /*keepdim=*/true) }, 0);
		norm_gate = ch_gate_pair.sub(mean);
		std = at::stack({ norm_gate[0].var(/*dim=*/1, /*unbiased=*/false, /*keepdim*/false),
						  norm_gate[1].slice(1, 0, state_size_3).var(/*dim=*/1, /*unbiased=*/false, /*keepdim*/false) }, 0).add(epsilon).sqrt();
		stds_collection.slice(0, 1, 3).select(1, i) = std;
		norm_gate /= std.unsqueeze(2);
		norm_gates_hh[i] = norm_gate[0];
		norm_gates_ch[i] = norm_gate[1].slice(1, 0, state_size_3);

		current_gate += norm_gate.mul(gammas_hh_ch).sum(0);

		current_gate.slice(1, 0, state_size_3).sigmoid_();
		current_gate.slice(1, state_size_3).tanh_();

		hc[1] = at::addcmul(current_gate.slice(1, state_size_3) * current_gate.slice(1, state_size, 2 * state_size), hc[1], current_gate.slice(1, 0, state_size));

		tanh_cell = at::tanh(hc[1]);
		tanh_new_cells[i] = tanh_cell;

		norm_tanh_cell = tanh_cell.sub(tanh_cell.mean(/*dim=*/1, /*keepdim=*/true));
		std = tanh_cell.var(/*dim=*/1, /*unbiased=*/false, /*keepdim=*/false).add(epsilon).sqrt();
		stds_collection[3][i] = std;
		norm_tanh_cell /= std.unsqueeze(1);
		norm_tanh_cells[i] = norm_tanh_cell;

		norm_tanh_cell = at::addcmul(beta_tanh_cell, norm_tanh_cell, gamma_tanh_cell);
		bnorm_tanh_cells[i] = norm_tanh_cell;

		//hc[0] = norm_tanh_cell * sig_gates[2];
		hc[0] = norm_tanh_cell * current_gate.slice(1, 2 * state_size, state_size_3);

		output[i] = hc[0];
	}

	// Output Dropout
	output *= dropout_output;

	return { output,
		hc[0],
		hc[1],
		norm_collection,
		tanh_new_cells,
		bnorm_tanh_cells,
		stds_collection,
		dropout,
		gates,
		X };
}

std::vector<at::Tensor> ln_peephole_lstm_layer_backward(
	at::Tensor grad_output,
	at::Tensor grad_h,
	at::Tensor grad_cell,
	at::Tensor norm_collection,
	at::Tensor tanh_new_cells,
	at::Tensor bnorm_tanh_cells,
	at::Tensor stds_collection,
	at::Tensor dropout,
	at::Tensor gates,
	at::Tensor X,
	at::Tensor weight_ih,
	at::Tensor weight_hh,
	at::Tensor weight_ch,
	at::Tensor gamma_ih,
	at::Tensor gamma_hh,
	at::Tensor gamma_ch,
	at::Tensor gamma_tanh_cell,
	bool training)
{
	const auto sequence_length = X.size(0);
	const auto batch_size = X.size(1);
	const auto input_size = weight_ih.size(1);
	const auto state_size = grad_h.size(1);
	const auto state_size_3 = 3 * state_size;
	const auto gate_size = state_size_3 + state_size;
	
	const bool state_ge_input = (state_size >= input_size);
	at::Tensor weights;
	if (state_ge_input)
	{
		if (state_size == input_size)
		{
			X = at::stack({ X.slice(2, 2 * state_size), X.slice(2, 0, state_size), X.slice(2, state_size, 2 * state_size) }, 0);
			weights = at::stack({ weight_ih, weight_hh, at::cat({ weight_ch, at::zeros(weight_ch.type(), { state_size, state_size }) }, 0) }, 0);
		}
		else
		{
			const int diff = state_size - input_size;
			X = at::stack({ at::cat({ X.slice(2, 2 * state_size), at::zeros(X.type(), { sequence_length, batch_size, diff }) }, 2),
							X.slice(2, 0, state_size),
							X.slice(2, state_size, 2 * state_size) }, 0);
			weights = at::stack({ at::cat({ weight_ih, at::zeros(weight_ih.type(), { gate_size, diff }) }, 1),
								  weight_hh,
								  at::cat({ weight_ch, at::zeros(weight_ch.type(), { state_size, state_size }) }, 0) }, 0);
		}
	}
	else
	{
		const int diff = input_size - state_size;
		X = at::cat({ X.slice(2, 2 * state_size).unsqueeze(0),
					  at::cat({ at::stack({ X.slice(2, 0, state_size),
											X.slice(2, state_size, 2 * state_size) }, 0),
					  at::zeros(X.type(), { 2, sequence_length, batch_size, diff })}, 3) }, 0);
		weights = at::cat({ weight_ih,
							at::cat({ at::stack({ weight_hh,
												 at::cat({ weight_ch, at::zeros(weight_ch.type(), { state_size, state_size }) }, 0) }, 0),
												 at::zeros(weight_hh.type(), { 2, gate_size, diff }) }, 2) }, 0);
	}

	const auto gammas_ihc = at::stack({ gamma_ih, gamma_hh, at::cat({ gamma_ch, at::zeros(gamma_ch.type(), { state_size }) }, 0) }, 0).unsqueeze(1);

	const auto norm_ihc = norm_collection;
	const auto norm_tanh_cells = norm_collection[2].slice(2, state_size_3);

	const auto dropout_hidden = dropout[0];
	const auto dropout_output = dropout[1];

	const auto forget_gates = gates.slice(2, 0, state_size);
	const auto output_gates = gates.slice(2, 2 * state_size, state_size_3);

	auto grad_inputs = at::zeros(X.type(), { sequence_length, batch_size, input_size });
	auto d_bnorm_tanh_new_cells = at::zeros_like(bnorm_tanh_cells);
	auto d_gates_ihc = at::zeros(gates.type(), { 3, sequence_length, batch_size, gate_size });

	grad_output *= dropout_output;

	gates = at::cat({ X[2],
					  gates.slice(/*dim=*/2, state_size_3),
					  bnorm_tanh_cells,
					  gates.slice(/*dim=*/2, state_size, 2 * state_size) }, /*dim=*/2)
		    * at::cat({ (gates.slice(/*dim=*/2, 0, state_size_3) * (1 - gates.slice(/*dim=*/2, 0, state_size_3))),
					    (1 - gates.slice(/*dim=*/2, state_size_3).pow(2)) }, /*dim=*/2);

	tanh_new_cells = 1 - tanh_new_cells.pow(2);

	at::Tensor d_bnorm_tanh_new_cell;
	at::Tensor d_norm_tanh_new_cell;
	at::Tensor current_std_tanh_new_cell;
	at::Tensor d_tanh_new_cell;
	at::Tensor d_new_cell;
	at::Tensor d_norm_gate;
	at::Tensor d_norm_gate_ihc;
	at::Tensor d_gate_ihc;
	at::Tensor d_X_ihc;

	stds_collection.slice(0, 0, 2) *= gate_size;
	stds_collection[2] *= state_size_3;
	stds_collection[3] *= state_size;
	stds_collection.unsqueeze_(3);
	const auto stds_ihc = stds_collection.slice(0, 0, 3);
	const auto stds_tanh_new_cell = stds_collection[3];
	auto sizes = at::zeros(stds_collection.type(), { 3, 1, 1 });
	sizes.slice(0, 0, 2) = gate_size;
	sizes[2] = state_size_3;
	for (int i = (sequence_length - 1); i >= 0; i--)
	{
		grad_h += grad_output[i];

		d_bnorm_tanh_new_cell = grad_h * output_gates[i];
		d_bnorm_tanh_new_cells[i] = d_bnorm_tanh_new_cell;

		d_norm_tanh_new_cell = d_bnorm_tanh_new_cell * gamma_tanh_cell;

		d_tanh_new_cell = (state_size * d_norm_tanh_new_cell
						   - d_norm_tanh_new_cell.sum(1, true)
						   - norm_tanh_cells[i] * (d_norm_tanh_new_cell * norm_tanh_cells[i]).sum(1, true)) / stds_tanh_new_cell[i];

		d_new_cell = at::addcmul(grad_cell, d_tanh_new_cell, tanh_new_cells[i]);

		d_norm_gate = at::cat({ d_new_cell, d_new_cell, grad_h, d_new_cell }, 1) * gates[i];
		gates[i] = d_norm_gate;

		d_norm_gate_ihc = d_norm_gate * gammas_ihc;

		d_gate_ihc = (sizes * d_norm_gate_ihc
					  - d_norm_gate_ihc.sum(2, true)
					  - norm_ihc.select(1, i) * (d_norm_gate_ihc * norm_ihc.select(1, i)).sum(2, true)) / stds_ihc.select(1, i);
		d_gates_ihc.select(1, i) = d_gate_ihc;

		d_X_ihc = at::matmul(d_gate_ihc, weights);

		grad_inputs[i] = d_X_ihc[0].slice(1, 0, input_size);
		grad_h = d_X_ihc[1].slice(1, 0, state_size) * dropout_hidden[i];
		grad_cell = at::addcmul(d_X_ihc[2].slice(1, 0, state_size), d_new_cell, forget_gates[i]);
	}

	const auto grad_weights_ihc = at::matmul(d_gates_ihc.flatten(1, 2).transpose(1, 2), X.flatten(1, 2));
	const auto grad_weight_ih = grad_weights_ihc[0].slice(1, 0, input_size);
	const auto grad_weight_hh = grad_weights_ihc[1].slice(1, 0, state_size);
	const auto grad_weight_ch = grad_weights_ihc[2].slice(0, 0, state_size_3).slice(1, 0, state_size);

	const auto grad_bias = gates.sum({ 0, 1 });

	const auto grad_gammas_ihc = gates.mul(norm_ihc).sum({ 1, 2 });
	const auto grad_gamma_ih = grad_gammas_ihc[0];
	const auto grad_gamma_hh = grad_gammas_ihc[1];
	const auto grad_gamma_ch = grad_gammas_ihc[2].slice(0, 0, state_size_3);
	const auto grad_gamma_tanh_cell = d_bnorm_tanh_new_cells.mul(norm_tanh_cells).sum({ 0, 1 });

	const auto grad_beta_tanh_cell = d_bnorm_tanh_new_cells.sum({ 0, 1 });

	return { grad_h,
			 grad_cell,
			 grad_inputs,
			 grad_weight_ih,
			 grad_weight_hh,
			 grad_weight_ch,
			 grad_bias,
			 grad_gamma_ih,
			 grad_gamma_hh,
			 grad_gamma_ch,
			 grad_gamma_tanh_cell,
			 grad_beta_tanh_cell };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("forward", &ln_peephole_lstm_layer_forward, "LN Peephole LSTM layer forward");
	m.def("backward", &ln_peephole_lstm_layer_backward, "LN Peephole LSTM layer backward");
}
