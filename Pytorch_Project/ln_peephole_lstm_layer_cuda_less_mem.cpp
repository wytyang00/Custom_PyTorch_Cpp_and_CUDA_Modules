#include <torch/torch.h>
#include <vector>

std::vector<at::Tensor> ln_peephole_lstm_layer_cpu_forward(
	at::Tensor const &input,
	at::Tensor const &weight_ih,
	at::Tensor const &weight_hh,
	at::Tensor const &weight_ch,
	at::Tensor const &bias,
	at::Tensor const &gamma_f,
	at::Tensor const &gamma_i,
	at::Tensor const &gamma_g,
	at::Tensor const &gamma_o,
	at::Tensor const &gamma_new_cell,
	at::Tensor const &beta_new_cell,
	at::Tensor const &hidden,
	at::Tensor const &cell,
	double const &epsilon,
	double const &dropout_p,
	bool const &dropout_on_output,
	bool const &training,
	int64_t const &sequence_length,
	int64_t const &batch_size,
	int64_t const &input_size,
	int64_t const &state_size,
	int64_t const &state_size_2,
	int64_t const &state_size_3,
	int64_t const &gate_size)
{
	const auto options = weight_ih.options();

	auto hiddens = at::empty({ sequence_length, batch_size, state_size }, options);
	auto cells = at::empty({ sequence_length + 1, batch_size, state_size }, options);

	auto gates_fig_stds = at::empty({ sequence_length, batch_size, 3, 1 }, options);
	auto gates_fig_normalized = at::empty({ sequence_length, batch_size, 3, state_size }, options);

	auto gates_o_stds = at::empty({ sequence_length, batch_size, 1 }, options);
	auto gates_o_normalized = at::empty({ sequence_length, batch_size, state_size }, options);

	auto new_cells_stds = at::empty({ sequence_length, batch_size, 1 }, options);
	auto new_cells_normalized = at::empty({ sequence_length, batch_size, state_size }, options);

	auto outputs = at::empty({ sequence_length, batch_size, state_size }, options);

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
	at::Tensor output_gate;
	at::Tensor std;

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

		hc.slice(1, 0, state_size) = output_gate * hc.slice(1, state_size).tanh();

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
		gates_fig_normalized,
		gates_fig_stds,
		gates_o_normalized,
		gates_o_stds,
		new_cells_normalized,
		new_cells_stds,
		dropout };
}

void backward_preparation(
	at::Tensor const &gates_fig_normalized,
	at::Tensor const &gates_o_normalized,
	at::Tensor const &cells,
	at::Tensor const &dropout_candidate_cells,
	at::Tensor const &gamma_f,
	at::Tensor const &gamma_i,
	at::Tensor const &gamma_g,
	at::Tensor const &gamma_o,
	at::Tensor const &bias,
	at::Tensor &forget_gates,
	at::Tensor &grad_gates_layer_normalized,
	at::Tensor &grad_new_cells_wrt_tanh_cells,
	int64_t const &state_size)
{
	auto gates = at::cat({ gates_o_normalized.unsqueeze(2), gates_fig_normalized }, 2)
		* at::stack({ gamma_o, gamma_f, gamma_i, gamma_g })
		+ bias.view({ 4, state_size });
	gates.slice(2, 0, 3).sigmoid_();
	gates.select(2, 3).tanh_();

	forget_gates = gates.select(2, 1);

	const auto tanh_cells = cells.tanh();
	grad_new_cells_wrt_tanh_cells = (1 - tanh_cells.pow(2)) * gates.select(2, 0);

	const auto d_sig = gates.slice(2, 0, 3) * (1 - gates.slice(2, 0, 3));
	gates.slice(2, 2) *= dropout_candidate_cells;

	grad_gates_layer_normalized = at::stack({ cells, gates.select(2, 3), gates.select(2, 2), tanh_cells })
		* at::cat({ d_sig.slice(2, 1, 3), (1 - gates.slice(2, 3).pow(2)), d_sig.slice(2, 0, 1) }, 2);
}

std::vector<at::Tensor> ln_peephole_lstm_layer_cpu_backward(
	at::Tensor &grad_output,
	at::Tensor &grad_hidden,
	at::Tensor &grad_cell,
	at::Tensor const &input,
	at::Tensor const &hiddens,
	at::Tensor const &cells,
	at::Tensor const &gates_fig_normalized,
	at::Tensor &gates_fig_stds,
	at::Tensor const &gates_o_normalized,
	at::Tensor &gates_o_stds,
	at::Tensor const &new_cells_normalized,
	at::Tensor &new_cells_stds,
	at::Tensor const &dropout,
	at::Tensor const &weight_ih,
	at::Tensor const &weight_hh,
	at::Tensor const &weight_ch,
	at::Tensor const &bias,
	at::Tensor const &gamma_f,
	at::Tensor const &gamma_i,
	at::Tensor const &gamma_g,
	at::Tensor const &gamma_o,
	at::Tensor const &gamma_new_cell)
{
	const auto sequence_length = input.size(0);
	const auto batch_size = input.size(1);
	const auto input_size = input.size(2);
	const auto state_size = hiddens.size(2);
	const auto state_size_2 = 2 * state_size;
	const auto state_size_3 = state_size_2 + state_size;
	const auto gate_size = state_size_3 + state_size;

	gates_fig_stds *= state_size;
	gates_o_stds *= state_size;
	new_cells_stds *= state_size;

	const auto dropout_candidate_cells = dropout[0];
	grad_output *= dropout[1];

	auto grad_inputs = at::zeros_like(input);

	const auto weights = at::cat({ weight_hh,
								   at::cat({ weight_ch.slice(0, 0, state_size).diag(),
											 weight_ch.slice(0, state_size, state_size_2).diag(),
											 at::zeros({ state_size_2, state_size }, weight_ch.options()) }),
								   weight_ih }, 1);
	const auto weight_co = weight_ch.slice(0, state_size_2);
	const auto gamma_fig = at::stack({ gamma_f, gamma_i, gamma_g });

	auto grad_new_cells = at::empty_like(new_cells_normalized);
	at::Tensor forget_gates;
	at::Tensor grad_gates_layer_normalized;
	at::Tensor grad_new_cells_wrt_tanh_cells;
	backward_preparation(
		gates_fig_normalized,
		gates_o_normalized,
		cells.slice(0, 0, sequence_length),
		dropout_candidate_cells,
		gamma_f,
		gamma_i,
		gamma_g,
		gamma_o,
		bias,
		forget_gates,
		grad_gates_layer_normalized,
		grad_new_cells_wrt_tanh_cells,
		state_size);
	auto grad_gates_layer_normalized_fig = grad_gates_layer_normalized.slice(2, 0, 3);
	auto grad_gates_layer_normalized_o = grad_gates_layer_normalized.select(2, 3);
	auto grad_gates_raw = at::empty_like(grad_gates_layer_normalized);
	auto grad_gates_raw_fig = grad_gates_raw.slice(2, 0, 3);
	auto grad_gates_raw_o = grad_gates_raw.select(2, 3);

	at::Tensor grad_output_gate_normalized;
	at::Tensor grad_output_gate_raw;
	at::Tensor grad_new_cell_normalized;
	at::Tensor grad_new_cell_raw;
	at::Tensor grad_gate_fig_normalized;
	at::Tensor grad_gate_fig_raw;
	at::Tensor grad_X;

	for (int i = (sequence_length - 1); i >= 0; i--)
	{
		grad_hidden += grad_output[i];

		grad_gates_layer_normalized_o[i] *= grad_hidden;

		grad_output_gate_normalized = grad_gates_layer_normalized_o[i] * gamma_o;
		grad_output_gate_raw = (state_size * grad_output_gate_normalized
								- grad_output_gate_normalized.sum(/*dim=*/1, /*keepdim=*/true)
								- gates_o_normalized[i] * (grad_output_gate_normalized * gates_o_normalized[i]).sum(/*dim=*/1, /*keepdim=*/true)).div(gates_o_stds[i]);
		grad_gates_raw_o[i] = grad_output_gate_raw;

		grad_cell += at::addcmul(grad_output_gate_raw * weight_co, grad_hidden, grad_new_cells_wrt_tanh_cells[i]);
		grad_new_cells[i] = grad_cell;

		grad_new_cell_normalized = grad_cell * gamma_new_cell;
		grad_new_cell_raw = (state_size * grad_new_cell_normalized
							 - grad_new_cell_normalized.sum(/*dim=*/1, /*keepdim=*/true)
							 - new_cells_normalized[i] * (grad_new_cell_normalized * new_cells_normalized[i]).sum(/*dim=*/1, /*keepdim=*/true)).div(new_cells_stds[i]);
		grad_gates_layer_normalized_fig[i] *= grad_new_cell_raw.unsqueeze(1);

		grad_gate_fig_normalized = grad_gates_layer_normalized_fig[i] * gamma_fig;

		grad_gates_raw_fig[i] = (state_size * grad_gate_fig_normalized
								 - grad_gate_fig_normalized.sum(/*dim=*/2, /*keepdim=*/true)
								 - gates_fig_normalized[i] * (grad_gate_fig_normalized * gates_fig_normalized[i]).sum(/*dim=*/2, /*keepdim=*/true)).div(gates_fig_stds[i]);

		grad_X = grad_gates_raw[i].view({ batch_size, gate_size }).mm(weights);

		grad_hidden = grad_X.slice(1, 0, state_size);
		grad_cell = at::addcmul(grad_X.slice(1, state_size, state_size_2), forget_gates[i], grad_new_cell_raw);
		grad_inputs[i] = grad_X.slice(1, state_size_2);
	}
	const auto flattened_grad_gates_raw = grad_gates_raw.view({ sequence_length * batch_size, gate_size });
	const auto grad_weight_ih_hh = flattened_grad_gates_raw.t().mm(at::cat({ input, hiddens }, 2).view({ sequence_length * batch_size, input_size + state_size }));
	const auto grad_weight_ch = at::cat({ flattened_grad_gates_raw.slice(1, 0, state_size_2),
										  flattened_grad_gates_raw.slice(1, state_size_3) }, 1).mul(at::cat({ cells.slice(0, 0, sequence_length).repeat({ 1, 1, 2 }),
																											  cells.slice(0, 1) }, 2).view({ sequence_length * batch_size, state_size_3 })).sum(/*dim=*/0, /*keepdim=*/false);
	const auto grad_bias = grad_gates_layer_normalized.sum(/*dim=*/{ 0, 1 }, /*keepdim=*/false).flatten();

	const auto grad_gammas = grad_gates_layer_normalized.mul(at::cat({ gates_fig_normalized, gates_o_normalized.unsqueeze(2) }, 2)).sum(/*dim=*/{ 0, 1 }, /*keepdim=*/false);

	const auto grad_gamma_new_cell = grad_new_cells.mul(new_cells_normalized).sum(/*dim=*/{ 0, 1 }, /*keepdim=*/false);
	const auto grad_beta_new_cell = grad_new_cells.sum(/*dim=*/{ 0, 1 }, /*keepdim=*/false);

	return { grad_inputs,
			 grad_weight_ih_hh.slice(1, 0, input_size).contiguous(),
			 grad_weight_ih_hh.slice(1, input_size).contiguous(),
			 grad_weight_ch,
			 grad_bias,
			 grad_gammas[0],
			 grad_gammas[1],
			 grad_gammas[2],
			 grad_gammas[3],
			 grad_gamma_new_cell,
			 grad_beta_new_cell,
			 grad_hidden,
			 grad_cell };
}

std::vector<at::Tensor> ln_peephole_lstm_layer_cuda_forward(
	at::Tensor const &input,
	at::Tensor const &weight_ih,
	at::Tensor const &weight_hh,
	at::Tensor const &weight_ch,
	at::Tensor const &bias,
	at::Tensor const &gamma_f,
	at::Tensor const &gamma_i,
	at::Tensor const &gamma_g,
	at::Tensor const &gamma_o,
	at::Tensor const &gamma_new_cell,
	at::Tensor const &beta_new_cell,
	at::Tensor &hidden,
	at::Tensor &cell,
	double const &epsilon,
	double const &dropout_p,
	bool const &dropout_on_output,
	bool const &training,
	int64_t const &sequence_length,
	int64_t const &batch_size,
	int64_t const &input_size,
	int64_t const &state_size,
	int64_t const &state_size_2,
	int64_t const &state_size_3,
	int64_t const &gate_size);

std::vector<at::Tensor> ln_peephole_lstm_layer_cuda_backward(
	at::Tensor &grad_output,
	at::Tensor &grad_hidden,
	at::Tensor &grad_cell,
	at::Tensor const &input,
	at::Tensor const &hiddens,
	at::Tensor const &cells,
	at::Tensor const &gates_fig_normalized,
	at::Tensor &gates_fig_stds,
	at::Tensor const &gates_o_normalized,
	at::Tensor &gates_o_stds,
	at::Tensor const &new_cells_normalized,
	at::Tensor &new_cells_stds,
	at::Tensor const &dropout,
	at::Tensor const &weight_ih,
	at::Tensor const &weight_hh,
	at::Tensor const &weight_ch,
	at::Tensor const &bias,
	at::Tensor const &gamma_f,
	at::Tensor const &gamma_i,
	at::Tensor const &gamma_g,
	at::Tensor const &gamma_o,
	at::Tensor const &gamma_new_cell);

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
	// Input dimension check (Confusion with input dimensions are quite usual. Please keep this check.)
	AT_ASSERTM(input.dim() == 3, "### The input tensor must have 3 dimensions, but the given tensor only has ", input.dim(), " dimension(s) ###");

	const auto sequence_length = input.size(0);
	const auto batch_size = input.size(1);
	const auto input_size = input.size(2);
	const auto state_size = weight_ih.size(0) / 4;
	const auto state_size_2 = state_size + state_size;
	const auto state_size_3 = state_size_2 + state_size;
	const auto gate_size = weight_ih.size(0);

	// Hiddens check (Frequent problems. Please, keep these checks.)
	AT_ASSERTM((hidden.dim() == 2) && (hidden.size(0) == batch_size) && (hidden.size(1) == state_size),
			   "### Invalid dimensions for the hidden state: ", hidden.sizes(), " Expected dimensions: ", at::IntList({ batch_size, state_size }), " ###");
	AT_ASSERTM((cell.dim() == 2) && (cell.size(0) == batch_size) && (cell.size(1) == state_size),
			   "### Invalid dimensions for the cell state: ", cell.sizes(), " Expected dimensions: ", at::IntList({ batch_size, state_size }), " ###");

	// Dropout check (Simple check, should be checked when creating a model as well. In that case, it might be unnecessary to do this test again here.)
	AT_ASSERTM((dropout_p >= 0) || (dropout_p <= 1), "### The value of dropout must be within the range of [0, 1], but ", dropout_p, " was given ###");

	// Device check (This part is important since this issue happens quite often)
	bool use_cuda = input.is_cuda();
	AT_ASSERTM((use_cuda == weight_ih.is_cuda()) && (use_cuda == weight_hh.is_cuda()) && (use_cuda == weight_ch.is_cuda())
			   && (use_cuda == bias.is_cuda()) && (use_cuda == gamma_f.is_cuda()) && (use_cuda == gamma_i.is_cuda()) && (use_cuda == gamma_g.is_cuda()) && (use_cuda == gamma_o.is_cuda())
			   && (use_cuda == gamma_new_cell.is_cuda()) && (use_cuda == beta_new_cell.is_cuda()) && (use_cuda == hidden.is_cuda()) && (use_cuda == cell.is_cuda()),
			   "### All tensors must be located in either CPU or CUDA devices together, but some of the given tensors are in a different device ###");

	if (false)
	{
		// Contiguity check (IMPORTANT FOR CUDA OPERATIONS; non-contiguous tensors result in irregular indexing and, therefore, calculation errors)
		AT_ASSERTM(input.is_contiguous(), "### input tensor is not contiguous ###");
		AT_ASSERTM(weight_ih.is_contiguous(), "### weight_ih tensor is not contiguous ###");
		AT_ASSERTM(weight_hh.is_contiguous(), "### weight_hh tensor is not contiguous ###");
		AT_ASSERTM(weight_ch.is_contiguous(), "### weight_ch tensor is not contiguous ###");
		AT_ASSERTM(bias.is_contiguous(), "### bias tensor is not contiguous ###");
		AT_ASSERTM(gamma_f.is_contiguous(), "### gamma_f tensor is not contiguous ###");
		AT_ASSERTM(gamma_i.is_contiguous(), "### gamma_i tensor is not contiguous ###");
		AT_ASSERTM(gamma_g.is_contiguous(), "### gamma_g tensor is not contiguous ###");
		AT_ASSERTM(gamma_o.is_contiguous(), "### gamma_o tensor is not contiguous ###");
		AT_ASSERTM(gamma_new_cell.is_contiguous(), "### gamma_new_cell tensor is not contiguous ###");
		AT_ASSERTM(beta_new_cell.is_contiguous(), "### beta_new_cell tensor is not contiguous ###");
		AT_ASSERTM(hidden.is_contiguous(), "### hidden tensor is not contiguous ###");
		AT_ASSERTM(cell.is_contiguous(), "### cell tensor is not contiguous ###");

		return ln_peephole_lstm_layer_cuda_forward(input,
												   weight_ih, weight_hh, weight_ch, bias, gamma_f, gamma_i, gamma_g, gamma_o, gamma_new_cell, beta_new_cell,
												   hidden, cell,
												   epsilon, dropout_p, dropout_on_output, training,
												   sequence_length, batch_size, input_size, state_size, state_size_2, state_size_3, gate_size);
	}
	else
	{
		return ln_peephole_lstm_layer_cpu_forward(input,
												  weight_ih, weight_hh, weight_ch, bias, gamma_f, gamma_i, gamma_g, gamma_o, gamma_new_cell, beta_new_cell,
												  hidden, cell,
												  epsilon, dropout_p, dropout_on_output, training,
												  sequence_length, batch_size, input_size, state_size, state_size_2, state_size_3, gate_size);
	}
}

std::vector<at::Tensor> ln_peephole_lstm_layer_backward(
	at::Tensor grad_output,
	at::Tensor grad_hidden,
	at::Tensor grad_cell,
	at::Tensor input,
	at::Tensor hiddens,
	at::Tensor cells,
	at::Tensor gates_fig_normalized,
	at::Tensor gates_fig_stds,
	at::Tensor gates_o_normalized,
	at::Tensor gates_o_stds,
	at::Tensor new_cells_normalized,
	at::Tensor new_cells_stds,
	at::Tensor dropout,
	at::Tensor weight_ih,
	at::Tensor weight_hh,
	at::Tensor weight_ch,
	at::Tensor bias,
	at::Tensor gamma_f,
	at::Tensor gamma_i,
	at::Tensor gamma_g,
	at::Tensor gamma_o,
	at::Tensor gamma_new_cell)
{
	// Not much checks since the values are saved during the forward pass and are supposed to be valid... just some device and contiguity checks
	bool use_cuda = grad_output.is_cuda();
	AT_ASSERTM((use_cuda == grad_hidden.is_cuda()) && (use_cuda == grad_cell.is_cuda()) && (use_cuda == input.is_cuda()) && (use_cuda == hiddens.is_cuda()) && (use_cuda == cells.is_cuda())
			   && (use_cuda == gates_fig_normalized.is_cuda()) && (use_cuda == gates_fig_stds.is_cuda())
			   && (use_cuda == gates_o_normalized.is_cuda()) && (use_cuda == gates_o_stds.is_cuda())
			   && (use_cuda == new_cells_normalized.is_cuda()) && (use_cuda == new_cells_stds.is_cuda()) && (use_cuda == dropout.is_cuda())
			   && (use_cuda == weight_ih.is_cuda()) && (use_cuda == weight_hh.is_cuda()) && (use_cuda == weight_ch.is_cuda()) && (use_cuda == bias.is_cuda())
			   && (use_cuda == gamma_f.is_cuda()) && (use_cuda == gamma_i.is_cuda()) && (use_cuda == gamma_g.is_cuda()) && (use_cuda == gamma_o.is_cuda()) && (use_cuda == gamma_new_cell.is_cuda()),
			   "### All tensors must be located in either CPU or CUDA devices together, but some of the given tensors are in a different device ###");

	if (false)
	{
		// Contiguity check
		AT_ASSERTM(grad_output.is_contiguous(), "### grad_output tensor is not contiguous ###");
		AT_ASSERTM(grad_hidden.is_contiguous(), "### grad_h tensor is not contiguous ###");
		AT_ASSERTM(grad_cell.is_contiguous(), "### grad_cell tensor is not contiguous ###");
		AT_ASSERTM(input.is_contiguous(), "### input tensor is not contiguous ###");
		AT_ASSERTM(hiddens.is_contiguous(), "### hiddens tensor is not contiguous ###");
		AT_ASSERTM(cells.is_contiguous(), "### cells tensor is not contiguous ###");
		AT_ASSERTM(gates_fig_normalized.is_contiguous(), "### gates_fig_normalized tensor is not contiguous ###");
		AT_ASSERTM(gates_fig_stds.is_contiguous(), "### gates_fig_stds tensor is not contiguous ###");
		AT_ASSERTM(gates_o_normalized.is_contiguous(), "### gates_o_normalized tensor is not contiguous ###");
		AT_ASSERTM(gates_o_stds.is_contiguous(), "### gates_o_stds tensor is not contiguous ###");
		AT_ASSERTM(new_cells_normalized.is_contiguous(), "### new_cells_normalized tensor is not contiguous ###");
		AT_ASSERTM(new_cells_stds.is_contiguous(), "### new_cells_stds tensor is not contiguous ###");
		AT_ASSERTM(dropout.is_contiguous(), "### dropout tensor is not contiguous ###");
		AT_ASSERTM(weight_ih.is_contiguous(), "### weight_ih tensor is not contiguous ###");
		AT_ASSERTM(weight_hh.is_contiguous(), "### weight_ih tensor is not contiguous ###");
		AT_ASSERTM(weight_ch.is_contiguous(), "### weight_ih tensor is not contiguous ###");
		AT_ASSERTM(bias.is_contiguous(), "### bias tensor is not contiguous ###");
		AT_ASSERTM(gamma_f.is_contiguous(), "### gamma_f tensor is not contiguous ###");
		AT_ASSERTM(gamma_i.is_contiguous(), "### gamma_i tensor is not contiguous ###");
		AT_ASSERTM(gamma_g.is_contiguous(), "### gamma_g tensor is not contiguous ###");
		AT_ASSERTM(gamma_o.is_contiguous(), "### gamma_o tensor is not contiguous ###");
		AT_ASSERTM(gamma_new_cell.is_contiguous(), "### gamma_tanh_cell tensor is not contiguous ###");

		return ln_peephole_lstm_layer_cuda_backward(grad_output, grad_hidden, grad_cell,
													input, hiddens, cells,
													gates_fig_normalized, gates_fig_stds,
													gates_o_normalized, gates_o_stds,
													new_cells_normalized, new_cells_stds,
													dropout,
													weight_ih, weight_hh, weight_ch, bias, gamma_f, gamma_i, gamma_g, gamma_o, gamma_new_cell);
	}
	else
	{
		return ln_peephole_lstm_layer_cpu_backward(grad_output, grad_hidden, grad_cell,
												   input, hiddens, cells,
												   gates_fig_normalized, gates_fig_stds,
												   gates_o_normalized, gates_o_stds,
												   new_cells_normalized, new_cells_stds,
												   dropout,
												   weight_ih, weight_hh, weight_ch, bias, gamma_f, gamma_i, gamma_g, gamma_o, gamma_new_cell);
	}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("forward", &ln_peephole_lstm_layer_forward, "LN Peephole LSTM layer forward (CUDA, less memory)");
	m.def("backward", &ln_peephole_lstm_layer_backward, "LN Peephole LSTM layer backward (CUDA, less memory)");
}
