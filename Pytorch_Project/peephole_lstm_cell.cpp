#include <torch/torch.h>
#include <vector>

std::vector<at::Tensor> peephole_lstm_cell_forward(
	at::Tensor const &input,
	at::Tensor const &weight_ih,
	at::Tensor const &weight_hh,
	at::Tensor const &weight_ch,
	at::Tensor const &bias,
	at::Tensor const &old_h,
	at::Tensor const &old_cell)
{
	const auto state_size = old_h.size(1);
	auto X = at::cat({ old_h, old_cell, input }, /*dim=*/1);
	auto weights = at::cat({ weight_hh, at::cat({ weight_ch, at::zeros(weight_ch.type(), { state_size, state_size }) }, 0), weight_ih }, 1);

	auto gate_weights = at::addmm(bias, X, weights.transpose(0, 1));

	auto sig_gates = gate_weights.slice(/*dim=*/1, 0, 3 * state_size).sigmoid();
	auto tanh_gate = gate_weights.slice(/*dim=*/1, 3 * state_size).tanh();

	auto new_cell = old_cell * sig_gates.slice(1, 0, state_size) + tanh_gate * sig_gates.slice(1, state_size, 2 * state_size);
	auto tanh_new_cell = new_cell.tanh();
	auto new_h = tanh_new_cell * sig_gates.slice(1, 2 * state_size);

	return { new_h,
		new_cell,
		tanh_new_cell,
		at::cat({ sig_gates, tanh_gate }, 1),
		X };
}

std::vector<at::Tensor> peephole_lstm_cell_backward(
	at::Tensor const &grad_h,
	at::Tensor const &grad_cell,
	at::Tensor const &old_cell,
	at::Tensor const &tanh_new_cell,
	at::Tensor const &gates,
	at::Tensor const &X,
	at::Tensor const &weight_ih,
	at::Tensor const &weight_hh,
	at::Tensor const &weight_ch)
{
	const auto state_size = grad_h.size(1);

	auto d_tanh_new_cell = gates.slice(1, 2 * state_size, 3 * state_size) * grad_h;
	auto d_new_cell = (1 - tanh_new_cell.pow(2)) * d_tanh_new_cell + grad_cell;

	auto d_old_cell = gates.slice(1, 0, state_size) * d_new_cell;

	auto d_gates = at::cat({
		gates.slice(1, 0, 3 * state_size) * ( 1 - gates.slice(1, 0, 3 * state_size) ),
		1 - gates.slice(1, 3 * state_size).pow(2)
						   }, 1);
	d_gates *= at::cat({ old_cell, gates.slice(1, 3 * state_size), tanh_new_cell, gates.slice(1, state_size, 2 * state_size) }, 1);
	d_gates *= at::cat({ d_new_cell, d_new_cell, grad_h, d_new_cell }, 1);

	auto d_weights = d_gates.t().mm(X);
	auto d_bias = d_gates.sum(/*dim=*/0, /*keepdim=*/false);
	auto d_X = d_gates.mm(at::cat({ weight_hh, at::cat({ weight_ch, at::zeros(weight_ch.type(), { state_size, state_size }) }, 0), weight_ih }, 1) );
	auto d_old_h = d_X.slice(/*dim=*/1, 0, state_size);
	d_old_cell += d_X.slice(/*dim=*/1, state_size, 2 * state_size);
	auto d_input = d_X.slice(/*dim=*/1, 2 * state_size);

	return { d_old_h, d_old_cell, d_input, d_weights.slice(1, 2 * state_size), d_weights.slice(1, 0, state_size), d_weights.slice(1, state_size, 2 * state_size).slice(0, 0, 3 * state_size), d_bias };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("forward", &peephole_lstm_cell_forward, "Peephole LSTM Cell forward");
	m.def("backward", &peephole_lstm_cell_backward, "Peephole LSTM Cell backward");
}
