#include <torch/torch.h>
#include <vector>

std::vector<at::Tensor> ln_peephole_lstm_layer_forward(
	at::Tensor input,
	at::Tensor weight_ih,
	at::Tensor weight_hh,
	at::Tensor weight_ch,
	at::Tensor bias,
	at::Tensor gamma_f,
	at::Tensor gamma_i,
	at::Tensor gamma_o,
	at::Tensor gamma_g,
	at::Tensor gamma_new_cell,
	at::Tensor beta_new_cell,
	at::Tensor hidden,
	at::Tensor cell,
	double epsilon,
	double dropout_p,
	bool dropout_output,
	bool training)
{
	AT_ASSERTM(input.dim() == 3, "### input must be a 3D tensor ###");
	const auto sequence_length = input.size(0);
	const auto batch_size = input.size(1);
	const auto input_size = input.size(2);
	const auto state_size = weight_ih.size(0) / 4;
	const auto state_size_2 = 2 * state_size;
	const auto state_size_3 = state_size_2 + state_size;
	const auto gate_size = state_size_3 + state_size;

	at::Tensor X = at::cat({ at::zeros(input.type(), { sequence_length, batch_size, state_size_2 }), input }, 2);

	at::Tensor norm_gate_collection = at::zeros(input.type(), { 4, sequence_length, batch_size, state_size });
	at::Tensor norm_fig = norm_gate_collection.slice(0, 0, 3);
	at::Tensor norm_output_gates = norm_gate_collection[3];

	at::Tensor stds_collection = at::zeros(norm_gate_collection.type(), { 4, sequence_length, batch_size, 1 });
	at::Tensor stds_fig = stds_collection.slice(0, 0, 3);
	at::Tensor stds_o = stds_collection[3];

	at::Tensor gates = at::matmul(input, weight_ih.transpose(0, 1));
	at::Tensor forget_gates = gates.slice(2, 0, state_size);
	at::Tensor input_gates = gates.slice(2, state_size, state_size_2);
	at::Tensor candidate_cells = gates.slice(2, state_size_2, state_size_3);
	at::Tensor output_gates= gates.slice(2, state_size_3);

	at::Tensor norm_new_cells = at::zeros(cell.type(), { sequence_length, batch_size, state_size });
	at::Tensor stds_new_cells = at::zeros(cell.type(), { sequence_length, batch_size, 1 });
	at::Tensor tanh_new_cells = at::zeros(cell.type(), { sequence_length, batch_size, state_size });

	at::Tensor output = at::zeros(weight_ih.type(), { sequence_length, batch_size, state_size });

	at::Tensor dropout;
	if (dropout_p <= 0. || !training) { dropout = at::ones(output.type(), { 2, sequence_length, batch_size, state_size }); }
	else
	{
		if (dropout_p >= 1.) { dropout = at::zeros(output.type(), { 2, sequence_length, batch_size, state_size }); }
		else { dropout = at::bernoulli(at::zeros(output.type(), { 2, sequence_length, batch_size, state_size }), (1 - dropout_p)).div(1 - dropout_p); }

		if (!dropout_output) { dropout[1] = 1; }
	}
	auto dropout_candidate_cell = dropout[0];
	auto dropout_output = dropout[1];

	/*auto hc = at::cat({ hidden, cell }, 1);
	hidden = hc.slice(1, 0, state_size);
	cell = hc.slice(1, state_size);*/

	weight_hh.t_();
	const auto weight_ch_fi = weight_ch.view({ 3, 1, state_size }).slice(0, 0, 2);
	const auto weight_ch_o = weight_ch.view({ 3, 1, state_size })[2];

	const auto bias_fig = bias.slice(0, 0, state_size_3);
	const auto bias_o = bias.slice(0, state_size_3);

	const auto fig_gammas = at::stack({ gamma_f, gamma_i, gamma_g }, 0).unsqueeze(1);

	at::Tensor std;
	at::Tensor current_gate;
	at::Tensor stacked_gate;
	at::Tensor tanh_new_cell;

	// Forward Loop
	for (int i = 0; i < sequence_length; i++)
	{
		current_gate = gates[i];

		//X[i].slice(1, 0, 2 * state_size) = hc;
		X[i].slice(1, 0, state_size) = hidden;
		X[i].slice(1, state_size, state_size_2) = cell;

		current_gate += at::mm(hidden, weight_hh);
		current_gate.slice(1, 0, state_size_2) += cell.mul(weight_ch_fi).permute({ 1, 0, 2 }).flatten(1, 2);

		//current_gate and stacked_gate share the same data
		stacked_gate = current_gate.view({ batch_size, 4, state_size });
		stacked_gate.slice(1, 0, 3) -= stacked_gate.slice(1, 0, 3).mean(/*dim=*/2, /*keepdim=*/true);
		std = stacked_gate.slice(1, 0, 3).var(/*dim=*/2, /*unbiased=*/false, /*keepdim=*/true).add(epsilon).sqrt();
		stds_fig.select(1, i) = std;
		stacked_gate.slice(1, 0, 3) /= std;
		norm_fig.select(1, i) = stacked_gate.slice(1, 0, 3);
		stacked_gate.slice(1, 0, 3) *= fig_gammas;
		current_gate.slice(1, 0, state_size_3) += bias_fig;

		current_gate.slice(1, 0, state_size_2).sigmoid_();
		current_gate.slice(1, state_size_2, state_size_3).tanh_();

		stacked_gate.select(1, 2) *= dropout_candidate_cell[i];

		cell = at::addcmul(stacked_gate.select(1, 1) * stacked_gate.select(1, 2), cell, stacked_gate.select(1, 0));

		current_gate.slice(1, state_size_3) += at::addcmul(bias_o, cell, weight_ch_o);

		stacked_gate.select(1, 3) -= stacked_gate.select(1, 3).mean(/*dim=*/1, /*keepdim=*/true);
		std = stacked_gate.select(1, 3).var(/*dim=*/1, /*unbiased=*/false, /*keepdim=*/true).add(epsilon).sqrt();
		stds_o[i] = std;
		stacked_gate.select(1, 3) /= std;
		norm_output_gates[i] = stacked_gate.select(1, 3);
		stacked_gate.select(1, 3) *= gamma_o;
		stacked_gate.select(1, 3) += bias_o;

		current_gate.slice(1, state_size_3).sigmoid_();

		std = cell.var(/*dim=*/1, /*unbiased=*/false, /*keepdim=*/true).add(epsilon).sqrt();
		stds_new_cells[i] = std;
		cell -= cell.mean(/*dim=*/1, /*keepdim=*/true);
		cell /= std;
		norm_new_cells[i] = cell;
		cell *= gamma_new_cell;
		cell += beta_new_cell;

		tanh_new_cell = at::tanh(cell);
		tanh_new_cells[i] = tanh_new_cell;

		hidden = stacked_gate.select(1, 3) * tanh_new_cell;

		output[i] = hidden;
	}

	// Output Dropout
	output *= dropout_output;

	return { output,
		hidden,
		cell,
		norm_gate_collection,
		stds_collection,
		norm_new_cells,
		stds_new_cells,
		tanh_new_cells,
		dropout,
		gates,
		X };
}

std::vector<at::Tensor> ln_peephole_lstm_layer_backward(
	at::Tensor grad_output,
	at::Tensor grad_hidden,
	at::Tensor grad_cell,
	at::Tensor cell,
	at::Tensor norm_gate_collection,
	at::Tensor stds_collection,
	at::Tensor norm_new_cells,
	at::Tensor stds_new_cells,
	at::Tensor tanh_new_cells,
	at::Tensor dropout,
	at::Tensor gates,
	at::Tensor X,
	at::Tensor weight_ih,
	at::Tensor weight_hh,
	at::Tensor weight_ch,
	at::Tensor gamma_f,
	at::Tensor gamma_i,
	at::Tensor gamma_o,
	at::Tensor gamma_g,
	at::Tensor gamma_new_cell)
{
	const auto sequence_length = X.size(0);
	const auto batch_size = X.size(1);
	const auto input_size = weight_ih.size(1);
	const auto state_size = grad_hidden.size(1);
	const auto state_size_2 = 2 * state_size;
	const auto state_size_3 = state_size_2 + state_size;
	const auto gate_size = state_size_3 + state_size;

	const auto dropout_candidate_cells = dropout[0];
	const auto dropout_output = dropout[1];

	const auto forget_gates = gates.slice(2, 0, state_size);
	const auto output_gates = gates.slice(2, state_size_3);

	const auto weight_co = weight_ch.slice(0, state_size_2);
	const auto weights_X = at::cat({ weight_hh,
								     at::cat({ weight_ch.slice(0, 0, state_size).diag(),
										       weight_ch.slice(0, state_size, state_size_2).diag(),
											   at::zeros(weight_ch.type(), { state_size_2, state_size })}, 0).t(),
								     weight_ih }, 1);

	const auto gamma_fig = at::stack({ gamma_f, gamma_i, gamma_g }, 0);

	auto grad_inputs = at::zeros(X.type(), { sequence_length, batch_size, input_size });
	auto grad_ln_new_cells = at::zeros(grad_cell.type(), { sequence_length, batch_size, state_size });
	auto grad_ln_gates = at::zeros(gates.type(), { sequence_length, batch_size, gate_size });

	grad_output *= dropout_output;

	gates = at::cat({ X.slice(/*dim=*/2, state_size, state_size_2),
					  gates.slice(/*dim=*/2, state_size_2, state_size_3) * dropout_candidate_cells,
					  gates.slice(/*dim=*/2, state_size, state_size_2) * dropout_candidate_cells,
					  tanh_new_cells }, /*dim=*/2)
		    * at::cat({ (gates.slice(/*dim=*/2, 0, state_size_2) * (1 - gates.slice(/*dim=*/2, 0, state_size_2))),
					    (1 - gates.slice(/*dim=*/2, state_size_2, state_size_3).pow(2)),
						(gates.slice(/*dim=*/2, state_size_3) * (1 - gates.slice(/*dim=*/2, state_size_3))) }, /*dim=*/2);

	tanh_new_cells = (1 - tanh_new_cells.pow(2)) * output_gates;

	at::Tensor current_gate;
	at::Tensor current_ln_gate;
	at::Tensor grad_ln_output_gate;
	at::Tensor grad_norm_output_gate;
	at::Tensor grad_raw_output_gate;
	at::Tensor grad_norm_new_cell;
	at::Tensor grad_raw_new_cell;
	at::Tensor grad_ln_fig_gate;
	at::Tensor grad_norm_fig_gate;
	at::Tensor grad_raw_fig_gate;
	at::Tensor grad_X;

	stds_collection *= state_size;
	stds_new_cells *= state_size;
	const auto stds_fig_gate = stds_collection.slice(0, 0, 3).permute({ 1, 2, 0, 3 });
	const auto stds_output_gate = stds_collection[3];

	const auto norm_fig_gates = norm_gate_collection.slice(0, 0, 3).permute({ 1, 2, 0, 3 });
	const auto norm_output_gates = norm_gate_collection[3];

	for (int i = (sequence_length - 1); i >= 0; i--)
	{
		current_gate = gates[i];
		current_ln_gate = grad_ln_gates[i];

		grad_hidden += grad_output[i];

		grad_ln_output_gate = grad_hidden * current_gate.slice(1, state_size_3);
		current_ln_gate.slice(1, state_size_3) = grad_ln_output_gate;

		grad_norm_output_gate = grad_ln_output_gate * gamma_o;

		grad_raw_output_gate = (state_size * grad_norm_output_gate
								- grad_norm_output_gate.sum(1, true)
								- norm_output_gates[i] * (grad_norm_output_gate * norm_output_gates[i]).sum(1, true)) / stds_output_gate[i];

		current_gate.slice(1, state_size_3) = grad_raw_output_gate;

		grad_cell += grad_raw_output_gate * weight_co + grad_hidden * tanh_new_cells[i];
		grad_ln_new_cells[i] = grad_cell;

		grad_norm_new_cell = grad_cell * gamma_new_cell;

		grad_raw_new_cell = (state_size * grad_norm_new_cell
							 - grad_norm_new_cell.sum(1, true)
							 - norm_new_cells[i] * (grad_norm_new_cell * norm_new_cells[i]).sum(1, true)) / stds_new_cells[i];

		grad_ln_fig_gate = grad_raw_new_cell * current_gate.slice(1, 0, state_size_3).view({ batch_size, 3, state_size });
		current_ln_gate.slice(1, 0, state_size_3) = grad_ln_fig_gate.flatten(1, 2);

		grad_norm_fig_gate = grad_ln_fig_gate * gamma_fig;

		grad_raw_fig_gate = (state_size * grad_norm_fig_gate
							 - grad_norm_fig_gate.sum(2, true)
							 - norm_fig_gates[i] * (grad_norm_fig_gate * norm_fig_gates[i]).sum(2, true)) / stds_fig_gate[i];
		current_gate.slice(1, 0, state_size_3) = grad_raw_fig_gate.flatten(1, 2);

		grad_X = current_gate.mm(weights_X);

		grad_hidden = grad_X.slice(1, 0, state_size);
		grad_cell = grad_X.slice(1, state_size, state_size_2) + grad_raw_new_cell * forget_gates[i];
		grad_inputs[i] = grad_X.slice(1, state_size_2);
	}

	const auto grad_weight_ih = at::mm(gates.flatten(0, 1).t(), X.slice(2, state_size_2).flatten(0, 1));
	const auto grad_weight_hh = at::mm(gates.flatten(0, 1).t(), X.slice(2, 0, state_size).flatten(0, 1));
	const auto grad_weight_ch = gates.slice(2, 0, state_size_2).mul(X.slice(2, state_size, state_size_2).repeat({ 1, 1, 2 })).sum({ 0, 1 })
							   + gates.slice(2, state_size_3).mul(at::cat({ X.slice(2, state_size, state_size_2).slice(0, 1), cell }, 0)).sum({ 0, 1 });

	const auto grad_bias = grad_ln_gates.sum({ 0, 1 });

	const auto grad_gammas_fico = grad_ln_gates.view({ sequence_length, batch_size, 4, state_size }).mul(norm_gate_collection.permute({ 1, 2, 0, 3 })).sum({ 0, 1 });
	const auto grad_gamma_new_cell = grad_ln_new_cells.mul(norm_new_cells).sum({ 0, 1 });

	const auto grad_beta_new_cell = grad_ln_new_cells.sum({ 0, 1 });

	return { grad_inputs,
			 grad_weight_ih,
			 grad_weight_hh,
			 grad_weight_ch,
			 grad_bias,
			 grad_gammas_fico[0],
			 grad_gammas_fico[1],
			 grad_gammas_fico[2],
			 grad_gammas_fico[3],
			 grad_gamma_new_cell,
		     grad_beta_new_cell,
		     grad_hidden,
			 grad_cell };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("forward", &ln_peephole_lstm_layer_forward, "LN Peephole LSTM layer forward");
	m.def("backward", &ln_peephole_lstm_layer_backward, "LN Peephole LSTM layer backward");
}
