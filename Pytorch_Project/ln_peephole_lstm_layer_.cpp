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
	at::Tensor gamma_g,
	at::Tensor gamma_o,
	at::Tensor gamma_new_cell,
	at::Tensor beta_new_cell,
	at::Tensor hidden,
	at::Tensor cell,
	double epsilon,
	double dropout_p,
	bool dropout_on_output,
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

	const auto options = weight_ih.options();

	auto hiddens = at::empty({ sequence_length, batch_size, state_size }, options);
	auto cells = at::empty({ sequence_length + 1, batch_size, state_size }, options);

	auto gates_fig_stds = at::empty({ sequence_length, batch_size, 3, 1 }, options);
	auto gates_fig_normalized = at::empty({ sequence_length, batch_size, 3, state_size }, options);
	auto gates_fig = at::empty({ sequence_length, batch_size, 3, state_size }, options);

	auto gates_o_stds = at::empty({ sequence_length, batch_size, 1 }, options);
	auto gates_o_normalized = at::empty({ sequence_length, batch_size, state_size }, options);
	auto gates_o = at::empty({ sequence_length, batch_size, state_size }, options);

	auto new_cells_stds = at::empty({ sequence_length, batch_size, 1 }, options);
	auto new_cells_normalized = at::empty({ sequence_length, batch_size, state_size }, options);

	auto tanh_new_cells = at::empty({ sequence_length, batch_size, state_size }, options);

	auto outputs = at::zeros({ sequence_length, batch_size, state_size }, options);

	at::Tensor dropout;
	if (dropout_p <= 0. || !training) { dropout = at::ones({ 2, sequence_length, batch_size, state_size }, options); }
	else
	{
		if (dropout_p >= 1.) { dropout = at::zeros({ 2, sequence_length, batch_size, state_size }, options); }
		else { dropout = at::bernoulli(at::zeros({ 2, sequence_length, batch_size, state_size }, options), (1 - dropout_p)).div(1 - dropout_p); }

		if (!dropout_on_output) { dropout[1] = 1; }
	}
	const auto dropout_candidate_cell = dropout[0];
	const auto dropout_output = dropout[1];

	const auto ih = at::matmul(input, weight_ih.t());

	auto hc = at::cat({ hidden, cell }, 1);
	const auto weight_hc_h = at::cat({ weight_hh.t(),
									   at::cat({ weight_ch.slice(0, 0, state_size).diag(),
											     weight_ch.slice(0, state_size, state_size_2).diag(),
												 at::zeros({ state_size_2, state_size }, options) }).t() });

	const auto weight_co = weight_ch.slice(0, state_size_2);

	const auto gamma_fig = at::stack({ gamma_f, gamma_i, gamma_g });

	const auto bias_fig = bias.slice(0, 0, state_size_3).view({ 3, state_size });
	const auto bias_o = bias.slice(0, state_size_3);

	at::Tensor current_gate;
	at::Tensor forget_gate;
	at::Tensor input_gate;
	at::Tensor candidate_cell;
	at::Tensor output_gate;
	at::Tensor std;
	at::Tensor tanh_cell;

	for (int i = 0; i < sequence_length; i++)
	{
		hiddens[i] = hc.slice(1, 0, state_size);
		cells[i] = hc.slice(1, state_size);

		current_gate = at::addmm(ih[i], hc, weight_hc_h).view({ batch_size, 4, state_size });
		current_gate.slice(1, 0, 3) -= current_gate.slice(1, 0, 3).mean(/*dim=*/2, /*keepdim=*/true);
		std = current_gate.slice(1, 0, 3).var(/*dim=*/2, /*unbiased=*/false, /*keepdim=*/true).add(epsilon).sqrt();
		gates_fig_stds[i] = std;
		current_gate.slice(1, 0, 3) /= std;
		gates_fig_normalized[i] = current_gate.slice(1, 0, 3);
		current_gate.slice(1, 0, 3) = at::addcmul(bias_fig, current_gate.slice(1, 0, 3), gamma_fig);
		current_gate.slice(1, 0, 2).sigmoid_();
		current_gate.select(1, 2).tanh_();
		current_gate.select(1, 2) *= dropout_candidate_cell[i];
		gates_fig[i] = current_gate.slice(1, 0, 3);

		hc.slice(1, state_size) = at::addcmul(hc.slice(1, state_size) * current_gate.select(1, 0), current_gate.select(1, 1), current_gate.select(1, 2));
		hc.slice(1, state_size) -= hc.slice(1, state_size).mean(/*dim=*/1, /*keepdim=*/true);
		std = hc.slice(1, state_size).var(/*dim=*/1, /*unbiased=*/false, /*keepdim=*/true).add(epsilon).sqrt();
		new_cells_stds[i] = std;
		hc.slice(1, state_size) /= std;
		new_cells_normalized[i] = hc.slice(1, state_size);
		hc.slice(1, state_size) = at::addcmul(beta_new_cell, hc.slice(1, state_size), gamma_new_cell);

		output_gate = at::addcmul(current_gate.select(1, 3), hc.slice(1, state_size), weight_co);
		output_gate -= output_gate.mean(/*dim=*/1, /*keepdim=*/true);
		std = output_gate.var(/*dim=*/1, /*unbiased=*/false, /*keepdim=*/true).add(epsilon).sqrt();
		gates_o_stds[i] = std;
		output_gate /= std;
		gates_o_normalized[i] = output_gate;
		output_gate = at::addcmul(bias_o, output_gate, gamma_o);
		output_gate.sigmoid_();
		gates_o[i] = output_gate;

		tanh_cell = hc.slice(1, state_size).tanh();
		tanh_new_cells[i] = tanh_cell;

		hc.slice(1, 0, state_size) = output_gate * tanh_cell;

		outputs[i] = hc.slice(1, 0, state_size);
	}
	cells[sequence_length] = hc.slice(1, state_size);

	// Output Dropout
	outputs *= dropout_output;

	return { outputs,
		hc.slice(1, 0, state_size).contiguous(),
		hc.slice(1, state_size).contiguous(),
		input,
		hiddens,
		cells,
		gates_fig,
		gates_fig_normalized,
		gates_fig_stds,
		gates_o,
		gates_o_normalized,
		gates_o_stds,
		new_cells_normalized,
		new_cells_stds,
		tanh_new_cells,
		dropout };
}

std::vector<at::Tensor> ln_peephole_lstm_layer_backward(
	at::Tensor grad_output,
	at::Tensor grad_hidden,
	at::Tensor grad_cell,
	at::Tensor input,
	at::Tensor hiddens,
	at::Tensor cells,
	at::Tensor gates_fig,
	at::Tensor gates_fig_normalized,
	at::Tensor gates_fig_stds,
	at::Tensor gates_o,
	at::Tensor gates_o_normalized,
	at::Tensor gates_o_stds,
	at::Tensor new_cells_normalized,
	at::Tensor new_cells_stds,
	at::Tensor tanh_new_cells,
	at::Tensor dropout,
	at::Tensor weight_ih,
	at::Tensor weight_hh,
	at::Tensor weight_ch,
	at::Tensor gamma_f,
	at::Tensor gamma_i,
	at::Tensor gamma_g,
	at::Tensor gamma_o,
	at::Tensor gamma_new_cell)
{
	const auto sequence_length = input.size(0);
	const auto batch_size = input.size(1);
	const auto input_size = input.size(2);
	const auto state_size = hiddens.size(2);
	const auto state_size_2 = 2 * state_size;
	const auto state_size_3 = state_size_2 + state_size;
	const auto gate_size = state_size_3 + state_size;

	const auto forget_gates = gates_fig.select(2, 0);
	const auto forget_gates_normalized = gates_fig_normalized.select(2, 0);
	const auto forget_gates_stds = gates_fig_stds.select(2, 0);
	const auto input_gates = gates_fig.select(2, 1);
	const auto input_gates_normalized = gates_fig_normalized.select(2, 1);
	const auto input_gates_stds = gates_fig_stds.select(2, 1);
	const auto candidate_cells = gates_fig.select(2, 2);
	const auto candidate_cells_normalized = gates_fig_normalized.select(2, 2);
	const auto candidate_cells_stds = gates_fig_stds.select(2, 2);
	const auto output_gates = gates_o;
	const auto output_gates_normalized = gates_o_normalized;
	const auto output_gates_stds = gates_o_stds;

	const auto dropout_candidate_cells = dropout[0];
	const auto dropout_output = dropout[1];

	const auto weight_if = weight_ih.slice(0, 0, state_size);
	const auto weight_ii = weight_ih.slice(0, state_size, state_size_2);
	const auto weight_ig = weight_ih.slice(0, state_size_2, state_size_3);
	const auto weight_io = weight_ih.slice(0, state_size_3);

	const auto weight_hf = weight_hh.slice(0, 0, state_size);
	const auto weight_hi = weight_hh.slice(0, state_size, state_size_2);
	const auto weight_hg = weight_hh.slice(0, state_size_2, state_size_3);
	const auto weight_ho = weight_hh.slice(0, state_size_3);

	const auto weight_cf = weight_ch.slice(0, 0, state_size);
	const auto weight_ci = weight_ch.slice(0, state_size, state_size_2);
	const auto weight_co = weight_ch.slice(0, state_size_2);

	auto grad_inputs = at::zeros_like(input);

	auto grad_weight_ih = at::zeros_like(weight_ih);
	auto grad_weight_if = grad_weight_ih.slice(0, 0, state_size);
	auto grad_weight_ii = grad_weight_ih.slice(0, state_size, state_size_2);
	auto grad_weight_ig = grad_weight_ih.slice(0, state_size_2, state_size_3);
	auto grad_weight_io = grad_weight_ih.slice(0, state_size_3);

	auto grad_weight_hh = at::zeros_like(weight_hh);
	auto grad_weight_hf = grad_weight_hh.slice(0, 0, state_size);
	auto grad_weight_hi = grad_weight_hh.slice(0, state_size, state_size_2);
	auto grad_weight_hg = grad_weight_hh.slice(0, state_size_2, state_size_3);
	auto grad_weight_ho = grad_weight_hh.slice(0, state_size_3);

	auto grad_weight_ch = at::zeros_like(weight_ch);
	auto grad_weight_cf = grad_weight_ch.slice(0, 0, state_size);
	auto grad_weight_ci = grad_weight_ch.slice(0, state_size, state_size_2);
	auto grad_weight_co = grad_weight_ch.slice(0, state_size_2);

	auto grad_bias = at::zeros({ gate_size }, weight_ih.options());
	auto grad_bias_f = grad_bias.slice(0, 0, state_size);
	auto grad_bias_i = grad_bias.slice(0, state_size, state_size_2);
	auto grad_bias_g = grad_bias.slice(0, state_size_2, state_size_3);
	auto grad_bias_o = grad_bias.slice(0, state_size_3);

	auto grad_gamma_f = at::zeros_like(gamma_f);
	auto grad_gamma_i = at::zeros_like(gamma_i);
	auto grad_gamma_g = at::zeros_like(gamma_g);
	auto grad_gamma_o = at::zeros_like(gamma_o);
	auto grad_gamma_new_cell = at::zeros_like(gamma_new_cell);
	auto grad_beta_new_cell = at::zeros({ state_size }, gamma_f.options());

	grad_output *= dropout_output;

	at::Tensor grad_output_gate;
	at::Tensor grad_tanh_new_cell;
	at::Tensor grad_output_gate_layer_normalized;
	at::Tensor grad_output_gate_normalized;
	at::Tensor grad_output_gate_raw;
	at::Tensor grad_new_cell_normalized;
	at::Tensor grad_new_cell_raw;
	at::Tensor grad_forget_gate;
	at::Tensor grad_forget_gate_layer_normalized;
	at::Tensor grad_forget_gate_normalized;
	at::Tensor grad_forget_gate_raw;
	at::Tensor grad_input_gate;
	at::Tensor grad_input_gate_layer_normalized;
	at::Tensor grad_input_gate_normalized;
	at::Tensor grad_input_gate_raw;
	at::Tensor grad_candidate_cell;
	at::Tensor grad_candidate_cell_layer_normalized;
	at::Tensor grad_candidate_cell_normalized;
	at::Tensor grad_candidate_cell_raw;
	
	for (int i = (sequence_length - 1); i >= 0; i--)
	{
		grad_hidden += grad_output[i];

		grad_output_gate = grad_hidden * tanh_new_cells[i];
		grad_tanh_new_cell = grad_hidden * output_gates[i];
		grad_cell += grad_tanh_new_cell * (1 - tanh_new_cells[i].pow(2));

		grad_output_gate_layer_normalized = grad_output_gate * (output_gates[i] * (1 - output_gates[i]));
		grad_bias_o += grad_output_gate_layer_normalized.sum(/*dim=*/0, /*keepdim=*/false);
		grad_gamma_o += grad_output_gate_layer_normalized.mul(output_gates_normalized[i]).sum(/*dim=*/0, /*keepdim=*/false);
		grad_output_gate_normalized = grad_output_gate_layer_normalized * gamma_o;
		grad_output_gate_raw = (state_size * grad_output_gate_normalized
								- grad_output_gate_normalized.sum(/*dim=*/1, /*keepdim=*/true)
								- output_gates_normalized[i] * (grad_output_gate_normalized * output_gates_normalized[i]).sum(/*dim=*/1, /*keepdim=*/true)).div(output_gates_stds[i] * state_size);

		grad_weight_io += grad_output_gate_raw.t().mm(input[i]);
		grad_weight_ho += grad_output_gate_raw.t().mm(hiddens[i]);
		grad_weight_co += grad_output_gate_raw.mul(cells[i + 1]).sum(/*dim=*/0, /*keepdim=*/false);
		grad_inputs[i] += grad_output_gate_raw.mm(weight_io);
		grad_hidden = grad_output_gate_raw.mm(weight_ho);
		grad_cell += grad_output_gate_raw.mul(weight_co);
		//AT_ASSERTM(false, "\n\nNo Prob\n\n");
		
		grad_beta_new_cell += grad_cell.sum(/*dim=*/0, /*keepdim=*/false);
		grad_gamma_new_cell += grad_cell.mul(new_cells_normalized[i]).sum(/*dim=*/0, /*keepdim=*/false);
		grad_new_cell_normalized = grad_cell * gamma_new_cell;
		grad_new_cell_raw = (state_size * grad_new_cell_normalized
							 - grad_new_cell_normalized.sum(/*dim=*/1, /*keepdim=*/true)
							 - new_cells_normalized[i] * (grad_new_cell_normalized * new_cells_normalized[i]).sum(/*dim=*/1, /*keepdim=*/true)).div(new_cells_stds[i] * state_size);

		grad_forget_gate = grad_new_cell_raw * cells[i];
		grad_cell = grad_new_cell_raw * forget_gates[i];
		grad_input_gate = grad_new_cell_raw * candidate_cells[i];
		grad_candidate_cell = grad_new_cell_raw * input_gates[i];

		grad_forget_gate_layer_normalized = grad_forget_gate * (forget_gates[i] * (1 - forget_gates[i]));
		grad_bias_f += grad_forget_gate_layer_normalized.sum(/*dim=*/0, /*keepdim=*/false);
		grad_gamma_f += grad_forget_gate_layer_normalized.mul(forget_gates_normalized[i]).sum(/*dim=*/0, /*keepdim=*/false);
		grad_forget_gate_normalized = grad_forget_gate_layer_normalized * gamma_f;
		grad_forget_gate_raw = (state_size * grad_forget_gate_normalized
								- grad_forget_gate_normalized.sum(/*dim=*/1, /*keepdim=*/true)
								- forget_gates_normalized[i] * (grad_forget_gate_normalized * forget_gates_normalized[i]).sum(/*dim=*/1, /*keepdim=*/true)).div(forget_gates_stds[i] * state_size);

		grad_weight_if += grad_forget_gate_raw.t().mm(input[i]);
		grad_weight_hf += grad_forget_gate_raw.t().mm(hiddens[i]);
		grad_weight_cf += grad_forget_gate_raw.mul(cells[i]).sum(/*dim=*/0, /*keepdim=*/false);
		grad_inputs[i] += grad_forget_gate_raw.mm(weight_if);
		grad_hidden += grad_forget_gate_raw.mm(weight_hf);
		grad_cell += grad_forget_gate_raw.mul(weight_cf);

		grad_input_gate_layer_normalized = grad_input_gate * (input_gates[i] * (1 - input_gates[i]));
		grad_bias_i += grad_input_gate_layer_normalized.sum(/*dim=*/0, /*keepdim=*/false);
		grad_gamma_i += grad_input_gate_layer_normalized.mul(input_gates_normalized[i]).sum(/*dim=*/0, /*keepdim=*/false);
		grad_input_gate_normalized = grad_input_gate_layer_normalized * gamma_i;
		grad_input_gate_raw = (state_size * grad_input_gate_normalized
							   - grad_input_gate_normalized.sum(/*dim=*/1, /*keepdim=*/true)
							   - input_gates_normalized[i] * (grad_input_gate_normalized * input_gates_normalized[i]).sum(/*dim=*/1, /*keepdim=*/true)).div(input_gates_stds[i] * state_size);

		grad_weight_ii += grad_input_gate_raw.t().mm(input[i]);
		grad_weight_hi += grad_input_gate_raw.t().mm(hiddens[i]);
		grad_weight_ci += grad_input_gate_raw.mul(cells[i]).sum(/*dim=*/0, /*keepdim=*/false);
		grad_inputs[i] += grad_input_gate_raw.mm(weight_ii);
		grad_hidden += grad_input_gate_raw.mm(weight_hi);
		grad_cell += grad_input_gate_raw.mul(weight_ci);

		grad_candidate_cell_layer_normalized = grad_candidate_cell * (1 - candidate_cells[i].pow(2));
		grad_bias_g += grad_candidate_cell_layer_normalized.sum(/*dim=*/0, /*keepdim=*/false);
		grad_gamma_g += grad_candidate_cell_layer_normalized.mul(candidate_cells_normalized[i]).sum(/*dim=*/0, /*keepdim=*/false);
		grad_candidate_cell_normalized = grad_candidate_cell_layer_normalized * gamma_g;
		grad_candidate_cell_raw = (state_size * grad_candidate_cell_normalized
								- grad_candidate_cell_normalized.sum(/*dim=*/1, /*keepdim=*/true)
								- candidate_cells_normalized[i] * (grad_candidate_cell_normalized * candidate_cells_normalized[i]).sum(/*dim=*/1, /*keepdim=*/true)).div(candidate_cells_stds[i] * state_size);

		grad_weight_ig += grad_candidate_cell_raw.t().mm(input[i]);
		grad_weight_hg += grad_candidate_cell_raw.t().mm(hiddens[i]);
		grad_inputs[i] += grad_candidate_cell_raw.mm(weight_ig);
		grad_hidden += grad_candidate_cell_raw.mm(weight_hg);
	}

	return { grad_inputs,
			 grad_weight_ih,
			 grad_weight_hh,
			 grad_weight_ch,
			 grad_bias,
			 grad_gamma_f,
			 grad_gamma_i,
			 grad_gamma_g,
			 grad_gamma_o,
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
