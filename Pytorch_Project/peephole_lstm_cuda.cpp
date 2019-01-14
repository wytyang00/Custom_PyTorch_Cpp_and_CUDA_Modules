#include <torch/torch.h>
#include <vector>

// CPU forward / backward definitions

std::vector<at::Tensor> peephole_lstm_cpu_forward(
	at::Tensor &input,
	at::Tensor &weight_ih,
	at::Tensor &weight_hh,
	at::Tensor &weight_ch,
	at::Tensor &bias,
	at::Tensor &hidden,
	at::Tensor &cell,
	double &dropout_p,
	bool &training,
	int64_t const &sequence_length,
	int64_t const &batch_size,
	int64_t const &input_size,
	int64_t const &state_size)
{
	// Outputs and variables containers
	at::Tensor outputs = at::empty(weight_ih.type(), { sequence_length, batch_size, state_size });
	at::Tensor tanh_new_cells = at::empty(cell.type(), { sequence_length, batch_size, state_size }); // tanh(cell) values are useful for backpropagation
	at::Tensor gates = at::matmul(input, weight_ih.transpose(0, 1)); // inputs are already provided, so they can be computed all at once at this time
	at::Tensor X = at::empty(weight_ih.type(), { sequence_length, batch_size, (input_size + (2 * state_size)) }); // inputs, hiddens, cells combined and stored
	X.slice(2, 2 * state_size) = input; // all inputs are already given

	const auto weight_hc_h_t = at::cat({ weight_hh, at::cat({ weight_ch, at::zeros(weight_ch.type(), { state_size, state_size }) }, 0) }, 1).transpose(0, 1); // convenient and efficient form for matrix multiplication

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

std::vector<at::Tensor> peephole_lstm_cpu_backward(
	at::Tensor &grad_output,
	at::Tensor &grad_h,
	at::Tensor &grad_cell,
	at::Tensor &tanh_new_cells,
	at::Tensor &dropout,
	at::Tensor &gates,
	at::Tensor &X,
	at::Tensor &weight_ih,
	at::Tensor &weight_hh,
	at::Tensor &weight_ch)
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

	return { grad_h, grad_cell, grad_inputs, d_weights.slice(1, 2 * state_size), d_weights.slice(1, 0, state_size), d_weights.slice(1, state_size, 2 * state_size).slice(0, 0, 3 * state_size), d_bias };
}

// CUDA forward / backward declarations

std::vector<at::Tensor> peephole_lstm_cuda_forward(
	at::Tensor const &input,
	at::Tensor const &weight_ih,
	at::Tensor const &weight_hh,
	at::Tensor const &weight_ch,
	at::Tensor const &bias,
	at::Tensor const &old_h,
	at::Tensor const &old_cell,
	double const &dropout_p,
	bool const &training,
	int64_t const &sequence_length,
	int64_t const &batch_size,
	int64_t const &input_size,
	int64_t const &state_size);

std::vector<at::Tensor> peephole_lstm_cuda_backward(
	at::Tensor &grad_output,
	at::Tensor &grad_h,
	at::Tensor &grad_cell,
	at::Tensor &tanh_new_cells,
	at::Tensor const &dropout,
	at::Tensor &gates,
	at::Tensor const &X,
	at::Tensor const &weight_ih,
	at::Tensor const &weight_hh,
	at::Tensor const &weight_ch);

// C++ interface

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
	// Input dimension check (Confusion with input dimensions are quite usual. Please keep this check.)
	AT_ASSERTM(input.dim() == 3, "### The input tensor must have 3 dimensions, but the given tensor only has ", input.dim(), " dimensions ###");

	const auto sequence_length = input.size(0);
	const auto batch_size = input.size(1);
	const auto input_size = input.size(2);

	// Weight & Bias validity check (This part might be unnecessary. If not needed, just comment this part out. *except for the state_size variable*)
	AT_ASSERTM(weight_ih.dim() == 2, "### weight_ih is not a 2D tensor, please check the dimensionality ###");
	AT_ASSERTM(weight_hh.dim() == 2, "### weight_hh is not a 2D tensor, please check the dimensionality ###");
	AT_ASSERTM(weight_ch.dim() == 2, "### weight_ch is not a 2D tensor, please check the dimensionality ###");
	AT_ASSERTM(bias.dim() == 1, "### bias is not a 1D tensor, please check the dimensionality ###");
	AT_ASSERTM(weight_ih.size(0) % 4 == 0, "### The feature dimension(dim 0) of weight_ih is invalid; the feature size must be 4 * state_size and, therefore, a multiple of 4 ###");
	const auto state_size = weight_ih.size(0) / 4;
	AT_ASSERTM((weight_ih.size(1) == input_size) && (weight_hh.size(0) == 4 * state_size) && (weight_hh.size(1) == state_size) && (weight_ch.size(0) == 3 * state_size) && (weight_ch.size(1) == state_size) && (bias.size(0) == 4 * state_size),
			   "### Invalid feature dimensions of weights and bias: ", weight_ih.sizes(), " ", weight_hh.sizes(), " ", weight_ch.sizes(), " ", bias.sizes(), ". With given weight_ih, ",
			   "the dimensions must be: ", at::IntList({ 4 * state_size, input_size }), " ", at::IntList({ 4 * state_size, state_size }), " ", at::IntList({ 3 * state_size, state_size }), " ", at::IntList({ 4 * state_size }), " ###");

	// Hiddens check (Frequent problems. Please, keep these checks.)
	AT_ASSERTM((old_h.dim() == 2) && (old_h.size(0) == batch_size) && (old_h.size(1) == state_size),
			   "### Invalid dimensions for the hidden state: ", old_h.sizes(), " Expected dimensions: ", at::IntList({ batch_size, state_size }), " ###");
	AT_ASSERTM((old_cell.dim() == 2) && (old_cell.size(0) == batch_size) && (old_cell.size(1) == state_size),
			   "### Invalid dimensions for the cell state: ", old_cell.sizes(), " Expected dimensions: ", at::IntList({ batch_size, state_size }), " ###");

	// Dropout check (Simple check, should be checked when creating a model as well. In that case, it might be unnecessary to do this test again here.)
	AT_ASSERTM((dropout_p >= 0) || (dropout_p <= 1), "### The value of dropout must be within the range of [0, 1], but ", dropout_p, " was given ###");

	// Device check (This part is important since this issue happens quite often)
	bool use_cuda = input.is_cuda();
	AT_ASSERTM((use_cuda == weight_ih.is_cuda()) && (use_cuda == weight_hh.is_cuda()) && (use_cuda == weight_ch.is_cuda())
			   && (use_cuda == bias.is_cuda()) && (use_cuda == old_h.is_cuda()) && (use_cuda == old_cell.is_cuda()),
			   "### All tensors must be located in either CPU or CUDA devices together, but some of the given tensors are in a different device ###");
	
	if (use_cuda)
	{
		// Contiguity check (IMPORTANT FOR CUDA OPERATIONS; non-contiguous tensors result in irregular indexing and, therefore, calculation errors)
		AT_ASSERTM(input.is_contiguous(), "### input tensor is not contiguous ###");
		AT_ASSERTM(weight_ih.is_contiguous(), "### weight_ih tensor is not contiguous ###");
		AT_ASSERTM(weight_hh.is_contiguous(), "### weight_hh tensor is not contiguous ###");
		AT_ASSERTM(weight_ch.is_contiguous(), "### weight_ch tensor is not contiguous ###");
		AT_ASSERTM(bias.is_contiguous(), "### bias tensor is not contiguous ###");
		AT_ASSERTM(old_h.is_contiguous(), "### old_h tensor is not contiguous ###");
		AT_ASSERTM(old_cell.is_contiguous(), "### old_cell tensor is not contiguous ###");

		return peephole_lstm_cuda_forward(input, weight_ih, weight_hh, weight_ch, bias, old_h, old_cell, dropout_p, training, sequence_length, batch_size, input_size, state_size);
	}
	else
	{
		return peephole_lstm_cpu_forward(input, weight_ih, weight_hh, weight_ch, bias, old_h, old_cell, dropout_p, training, sequence_length, batch_size, input_size, state_size);
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
	at::Tensor weight_ih,
	at::Tensor weight_hh,
	at::Tensor weight_ch)
{
	// Not much checks since the values are saved during the forward pass and are supposed to be valid... just some device and contiguity checks
	bool use_cuda = grad_output.is_cuda();
	AT_ASSERTM((use_cuda == grad_h.is_cuda()) && (use_cuda == grad_cell.is_cuda()) && (use_cuda == tanh_new_cells.is_cuda()) && (use_cuda == dropout.is_cuda())
			   && (use_cuda == gates.is_cuda()) && (use_cuda == X.is_cuda()) && (use_cuda == weight_ih.is_cuda()) && (use_cuda == weight_hh.is_cuda()) && (use_cuda == weight_ch.is_cuda()),
			   "### All tensors must be located in either CPU or CUDA devices together, but some of the given tensors are in a different device ###");

	if (use_cuda)
	{
		// Contiguity check
		AT_ASSERTM(grad_output.is_contiguous(), "### grad_output tensor is not contiguous ###");
		AT_ASSERTM(grad_h.is_contiguous(), "### grad_h tensor is not contiguous ###");
		AT_ASSERTM(grad_cell.is_contiguous(), "### grad_cell tensor is not contiguous ###");
		AT_ASSERTM(tanh_new_cells.is_contiguous(), "### tanh_new_cells tensor is not contiguous ###");
		AT_ASSERTM(dropout.is_contiguous(), "### dropout tensor is not contiguous ###");
		AT_ASSERTM(gates.is_contiguous(), "### gates tensor is not contiguous ###");
		AT_ASSERTM(X.is_contiguous(), "### X tensor is not contiguous ###");
		AT_ASSERTM(weight_ih.is_contiguous(), "### weight_ih tensor is not contiguous ###");
		AT_ASSERTM(weight_hh.is_contiguous(), "### weight_ih tensor is not contiguous ###");
		AT_ASSERTM(weight_ch.is_contiguous(), "### weight_ih tensor is not contiguous ###");

		return peephole_lstm_cuda_backward(grad_output, grad_h, grad_cell, tanh_new_cells, dropout, gates, X, weight_ih, weight_hh, weight_ch);
	}
	else
	{
		return peephole_lstm_cpu_backward(grad_output, grad_h, grad_cell, tanh_new_cells, dropout, gates, X, weight_ih, weight_hh, weight_ch);
	}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("forward", &peephole_lstm_forward, "Peephole LSTM forward (CUDA)");
	m.def("backward", &peephole_lstm_backward, "Peephole LSTM backward (CUDA)");
}
