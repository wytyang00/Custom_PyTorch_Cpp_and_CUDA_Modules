#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid(scalar_t const &z)
{
	return 1.0 / (1.0 + exp(-z));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t tanh(scalar_t const &z)
{
	auto exp_n2z = exp(-2 * z);
	return (1.0 - exp_n2z) / (1.0 + exp_n2z);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_sigmoid_with_output(scalar_t const &a)
{
	return a * (1.0 - a);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_tanh_with_output(scalar_t const &a)
{
	return 1.0 - (a * a);
}

template <typename scalar_t>
__global__ void gates_activation(
	scalar_t* __restrict__ gates,
	const size_t gates_first_idx,
	const size_t batch_size,
	const size_t gate_size,
	const size_t sig_size)
{
	const int column = blockIdx.x * blockDim.x + threadIdx.x;
	if (column < gate_size)
	{
		const int batch = blockIdx.y * blockDim.y + threadIdx.y;
		if (batch < batch_size)
		{
			const int index = gates_first_idx + batch * gate_size + column;

			if (column < sig_size) { gates[index] = sigmoid(gates[index]); }
			else { gates[index] = tanh(gates[index]); }
		}
	}
}

template <typename scalar_t>
__global__ void peephole_lstm_cuda_forward_kernel(
	scalar_t* __restrict__ X,
	const scalar_t* __restrict__ gates,
	const scalar_t* __restrict__ dropout_hidden,
	scalar_t* __restrict__ hc,
	scalar_t* __restrict__ output,
	scalar_t* __restrict__ tanh_new_cells,
	const size_t X_first_idx,
	const size_t gates_first_idx,
	const size_t tanh_new_cells_and_output_first_idx,
	const size_t batch_size,
	const size_t gate_size,
	const size_t state_size,
	const size_t X_size)
{
	const int column = blockIdx.x * blockDim.x + threadIdx.x;
	if (column < state_size)
	{
		const int batch = blockIdx.y * blockDim.y + threadIdx.y;
		if (batch < batch_size)
		{
			const int h_index = batch * 2 * state_size + column;
			const int c_index = h_index + state_size;

			const int X_h_index = X_first_idx + batch * X_size + column;
			const int X_c_index = X_h_index + state_size;

			const int gates_forget_index = gates_first_idx + batch * gate_size + column;
			const int gates_input_index = gates_forget_index + state_size;
			const int gates_output_index = gates_input_index + state_size;
			const int gates_candidate_index = gates_output_index + state_size;

			const int tanh_new_cells_and_output_and_dropout_index = tanh_new_cells_and_output_first_idx + batch * state_size + column;

			hc[h_index] *= dropout_hidden[tanh_new_cells_and_output_and_dropout_index];

			X[X_h_index] = hc[h_index];
			X[X_c_index] = hc[c_index];

			hc[c_index] = hc[c_index] * gates[gates_forget_index] + gates[gates_candidate_index] * gates[gates_input_index];
			tanh_new_cells[tanh_new_cells_and_output_and_dropout_index] = tanh(hc[c_index]);
			hc[h_index] = tanh_new_cells[tanh_new_cells_and_output_and_dropout_index] * gates[gates_output_index];
			output[tanh_new_cells_and_output_and_dropout_index] = hc[h_index];
		}
	}
}

template <typename scalar_t>
__global__ void process_dgates_dtanh_dout(
	const scalar_t* __restrict__ X,
	const scalar_t* __restrict__ dropout,
	const scalar_t* __restrict__ input_gates,
	const scalar_t* __restrict__ candidate_cells,
	scalar_t* __restrict__ gates,
	scalar_t* __restrict__ tanh_new_cells,
	scalar_t* __restrict__ d_output,
	const size_t batch_x_X,
	const size_t batch_x_gate,
	const size_t batch_x_state,
	const size_t batch_size,
	const size_t X_size,
	const size_t state_size,
	const size_t gate_size,
	const size_t total_sequence_length)
{
	/* PARTS TO BE PARALLELIZED
	(things to be passed in - X, gates, tanh_new_cells, d_gates_mult, batch_x_X, batch_x_gate, batch_x_state, X_size, state_size, gate_size, total_sequence_length)
	(things to be modified - tanh_new_cells, d_gates_mult)

	auto d_gates_mult = at::cat(
		{
			X.slice(dim=2, state_size, 2 * state_size),    // old_cell                         old_cell * sigmoid
			gates.slice(dim=2, 3 * state_size),            // candidate_gate                   candidate * sigmoid
			tanh_new_cells,                                // tanh_new_cells                   tanh_cell * sigmoid
			gates.slice(dim=2, state_size, 2 * state_size) // input_gate                       input * tanh
		}, dim=2);

	d_gates_mult *= at::cat({ (gates.slice(dim=2, 0, 3 * state_size) * (1 - gates.slice(dim=2, 0, 3 * state_size))), // sig_gates_derivatives
							  (1 - gates.slice(dim=2, 3 * state_size).pow(2)) }, dim=2);                             // tanh_gates_derivatives

	tanh_new_cells = (1 - tanh_new_cells.pow(2)); // replace tanh_new_cells with their derivatives

	at::cat({ X.slice(2, state_size, 2 * state_size),
		      gates.slice(2, 3 * state_size),
	          tanh_new_cells,
	          gates.slice(2, state_size, 2 * state_size)}, 2)
    * at::cat({ (gates.slice(2, 0, 3 * state_size) * (1 - gates.slice(2, 0, 3 * state_size))),
			(1 - gates.slice(2, 3 * state_size).pow(2)) }, 2);
	*/
	const int column = blockIdx.x * blockDim.x + threadIdx.x;
	if (column < gate_size + state_size)
	{
		const int batch = blockIdx.y * blockDim.y + threadIdx.y;
		if (batch < batch_size)
		{
			const int sequence = blockIdx.z * blockDim.z + threadIdx.z;
			if (sequence < total_sequence_length)
			{
				if (column < gate_size)
				{
					const int d_gates_index = batch_x_gate * sequence + gate_size * batch + column;
					if (column < state_size)
					{
						const int old_cell_index = batch_x_X * sequence + X_size * batch + state_size + column;
						gates[d_gates_index] = X[old_cell_index] * d_sigmoid_with_output(gates[d_gates_index]);
					}
					else { if (column < 2 * state_size)
					{
						const int candidate_cell_index = batch_x_state * sequence + state_size * batch + column - state_size;
						gates[d_gates_index] = candidate_cells[candidate_cell_index] * d_sigmoid_with_output(gates[d_gates_index]); // SOLVED using values from different position: candidate (while being modified in parallel)
					}
					else { if (column < 3 * state_size)
					{
						const int tanh_new_cell_index = batch_x_state * sequence + state_size * batch - 2 * state_size + column;
						const auto output_gate = gates[d_gates_index];
						gates[d_gates_index] = tanh_new_cells[tanh_new_cell_index] * d_sigmoid_with_output(output_gate);
						tanh_new_cells[tanh_new_cell_index] = d_tanh_with_output(tanh_new_cells[tanh_new_cell_index]) * output_gate;
					}
					else //if (column < 4 * state_size)
					{
						const int input_gate_index = batch_x_state * sequence + state_size * batch + column - 3 * state_size;
						gates[d_gates_index] = input_gates[input_gate_index] * d_tanh_with_output(gates[d_gates_index]); // SOLVED using values from different position: input (while being modified in parallel)
					}}}
				}
				else //if (column < 5 * state_size)
				{
					const int output_and_dropout_index = batch_x_state * sequence + state_size * batch - 4 * state_size + column;
					d_output[output_and_dropout_index] *= dropout[output_and_dropout_index];
				}
			}
		}
	}
}

template <typename scalar_t>
__global__ void peephole_lstm_cuda_backward_kernel(
	const scalar_t* __restrict__ forget_gates,
	const scalar_t* __restrict__ d_outputs,
	const scalar_t* __restrict__ d_cell_copy,
	scalar_t* __restrict__ d_h,
	scalar_t* __restrict__ d_cell,
	const scalar_t* __restrict__ d_tanh_new_cells,
	scalar_t* __restrict__ d_gates,
	const size_t batch_x_gate,
	const size_t batch_x_state,
	const int sequence_index,
	const size_t batch_size,
	const size_t state_size,
	const size_t gate_size)
{
	/*
	d_h = (d_h + grad_output);

	auto d_new_cell = d_tanh_of_new_cell * d_h + d_cell;

	d_cell = forget_gate * d_new_cell;

	auto d_gates = d_gates * at::cat({ d_new_cell, d_new_cell, d_h, d_new_cell }, dim=1);
	*/
	const int column = blockIdx.x * blockDim.x + threadIdx.x;
	if (column < gate_size)
	{
		const int batch = blockIdx.y * blockDim.y + threadIdx.y;
		if (batch < batch_size)
		{
			const int d_gates_index = batch_x_gate * sequence_index + gate_size * batch + column;
			if (column < state_size)
			{
				const int state_index = batch_x_state * sequence_index + state_size * batch + column;
				const int hidden_index = state_size * batch + column;
				d_gates[d_gates_index] *= d_tanh_new_cells[state_index] * (d_h[hidden_index] + d_outputs[state_index]) + d_cell_copy[hidden_index];
			}
			else { if (column < 2 * state_size)
			{
				const int state_index = batch_x_state * sequence_index + state_size * batch + column - state_size;
				const int hidden_index = state_size * batch + column - state_size;
				const auto d_new_cell = d_tanh_new_cells[state_index] * (d_h[hidden_index] + d_outputs[state_index]) + d_cell_copy[hidden_index];
				d_gates[d_gates_index] *= d_new_cell;
				d_cell[hidden_index] = forget_gates[state_index] * d_new_cell;
			}
			else { if (column < 3 * state_size)
			{
				const int state_index = batch_x_state * sequence_index + state_size * batch + column - 2 * state_size;
				const int hidden_index = state_size * batch + column - 2 * state_size;
				d_gates[d_gates_index] *= d_h[hidden_index] + d_outputs[state_index];
			}
			else //if (column < 4 * state_size)
			{
				const int state_index = batch_x_state * sequence_index + state_size * batch + column - 3 * state_size;
				const int hidden_index = state_size * batch + column - 3 * state_size;
				d_gates[d_gates_index] *= d_tanh_new_cells[state_index] * (d_h[hidden_index] + d_outputs[state_index]) + d_cell_copy[hidden_index];
			}}}
		}
	}
}

template <typename scalar_t>
__global__ void grad_in_h_c_update(
	const scalar_t* __restrict__ grad_X,
	const scalar_t* __restrict__ dropout,
	scalar_t* __restrict__ grad_h,
	scalar_t* __restrict__ grad_cell,
	scalar_t* __restrict__ grad_input,
	const size_t batch_x_state,
	const size_t batch_x_input,
	const int sequence_index,
	const size_t input_size,
	const size_t batch_size,
	const size_t state_size,
	const size_t X_size)
{
	const int column = blockIdx.x * blockDim.x + threadIdx.x;
	if (column < X_size)
	{
		const int batch = blockIdx.y * blockDim.y + threadIdx.y;
		if (batch < batch_size)
		{
			const int grad_X_index = batch * X_size + column;
			if (column < state_size)
			{
				const int grad_h_index = batch * state_size + column;
				const int dropout_index = sequence_index * batch_x_state + batch * state_size + column;
				grad_h[grad_h_index] = grad_X[grad_X_index] * dropout[dropout_index];
			}
			else { if (column < 2 * state_size)
			{
				const int grad_cell_index = batch * state_size + column - state_size;
				grad_cell[grad_cell_index] += grad_X[grad_X_index];
			}
			else //if (column < X_size)
			{
				const int grad_input_index = sequence_index * batch_x_input + batch * input_size + column - 2 * state_size;
				grad_input[grad_input_index] = grad_X[grad_X_index];
			}}
		}
	}
}

std::vector<at::Tensor> ln_peephole_lstm_layer_cuda_forward(
	at::Tensor const &input,
	at::Tensor const &weight_ih,
	at::Tensor const &weight_hh,
	at::Tensor const &weight_ch,
	at::Tensor const &bias,
	at::Tensor const &gamma_ih,
	at::Tensor const &gamma_hh,
	at::Tensor const &gamma_ch,
	at::Tensor const &gamma_tanh_cell,
	at::Tensor const &beta_tanh_cell,
	at::Tensor &hidden,
	at::Tensor &cell,
	double const &epsilon,
	double const &dropout_p,
	bool const &training,
	int64_t const &sequence_length,
	int64_t const &batch_size,
	int64_t const &input_size,
	int64_t const &state_size,
	int64_t const &state_size_3,
	int64_t const &gate_size)
{
	/*
		at::Tensor output = at::zeros(weight_ih.type(), { sequence_length, batch_size, state_size });

		at::Tensor tanh_new_cells = at::zeros(cell.type(), { sequence_length, batch_size, state_size });
		at::Tensor lnorm_tanh_cells = at::zeros_like(tanh_new_cells);

		at::Tensor norm_collection = at::zeros(input.type(), { 3, sequence_length, batch_size, gate_size });
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

		gates -= gates.mean(2, true);
		std = gates.var(2, false, false).add(epsilon).sqrt();
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

			mean = at::stack({ ch_gate_pair[0].mean(1, true),
							   ch_gate_pair[1].slice(1, 0, state_size_3).mean(1, true) }, 0);
			norm_gate = ch_gate_pair.sub(mean);
			std = at::stack({ norm_gate[0].var(1, false, false),
							  norm_gate[1].slice(1, 0, state_size_3).var(1, false, false) }, 0).add(epsilon).sqrt();
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

			norm_tanh_cell = tanh_cell.sub(tanh_cell.mean(1, true));
			std = tanh_cell.var(1, false, false).add(epsilon).sqrt();
			stds_collection[3][i] = std;
			norm_tanh_cell /= std.unsqueeze(1);
			norm_tanh_cells[i] = norm_tanh_cell;

			norm_tanh_cell = at::addcmul(beta_tanh_cell, norm_tanh_cell, gamma_tanh_cell);
			lnorm_tanh_cells[i] = norm_tanh_cell;

			//hc[0] = norm_tanh_cell * sig_gates[2];
			hc[0] = norm_tanh_cell * current_gate.slice(1, 2 * state_size, state_size_3);

			output[i] = hc[0];

		output *= dropout_output;
		*/
	at::Tensor output = at::zeros(weight_ih.type(), { sequence_length, batch_size, state_size });
	at::Tensor tanh_new_cells = at::zeros(cell.type(), { sequence_length, batch_size, state_size });
	at::Tensor lnorm_tanh_cells = at::zeros_like(tanh_new_cells);

	at::Tensor norm_collection = at::zeros(input.type(), { 3, sequence_length, batch_size, gate_size });
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

	AT_DISPATCH_FLOATING_TYPES(gates.type(), "peephole_lstm_cuda_forward", ([&] {
		for (int i = 0; i < sequence_length; i++)
		{
			//X[i].slice(1, 0, hc_size) = hc;
			gates[i] += at::addmm(bias, hc, weight_hc_h_t);
			
			gates_activation<scalar_t> <<<blocks_gates, threads>>> (
				gates.data<scalar_t>(),
				gates_first_idx,
				batch_size,
				gate_size,
				sig_size);
				
			peephole_lstm_cuda_forward_kernel<scalar_t> <<<blocks_update, threads>>> (
				X.data<scalar_t>(),
				gates.data<scalar_t>(),
				dropout_hidden.data<scalar_t>(),
				hc.data<scalar_t>(),
				outputs.data<scalar_t>(),
				tanh_new_cells.data<scalar_t>(),
				X_first_idx,
				gates_first_idx,
				tanh_new_cells_and_output_first_idx,
				batch_size,
				gate_size,
				state_size,
				X_size);
			
			X_first_idx += batch_x_X;
			gates_first_idx += batch_x_gate;
			tanh_new_cells_and_output_first_idx += batch_x_state;
		}
	}));

	outputs *= dropout_output;

	return { outputs,
		hc.slice(1, 0, state_size),
		hc.slice(1, state_size),
		tanh_new_cells,
		dropout,
		gates,
		X };
}

std::vector<at::Tensor> ln_peephole_lstm_layer_cuda_backward(
		at::Tensor &grad_output,
		at::Tensor &grad_h,
		at::Tensor &grad_cell,
		at::Tensor const &norm_collection,
		at::Tensor &tanh_new_cells,
		at::Tensor const &lnorm_tanh_cells,
		at::Tensor &stds_collection,
		at::Tensor const &dropout,
		at::Tensor &gates,
		at::Tensor &X,
		at::Tensor const &weight_ih,
		at::Tensor const &weight_hh,
		at::Tensor const &weight_ch,
		at::Tensor const &gamma_ih,
		at::Tensor const &gamma_hh,
		at::Tensor const &gamma_ch,
		at::Tensor const &gamma_tanh_cell)
{
	const auto sequence_length = X.size(0);
	const auto batch_size = X.size(1);
	const auto state_size = grad_h.size(1);
	const auto X_size = X.size(2);
	const auto input_size = X_size - (2 * state_size);

	const auto gate_size = gates.stride(1); // 4 * state_size;
	const auto batch_x_input = batch_size * input_size;
	const auto batch_x_state = dropout.stride(1); // batch_size * state_size;
	const auto batch_x_X = X.stride(0); // batch_x_input + 2 * batch_x_state;
	const auto batch_x_gate = gates.stride(0); // batch_x_state * 4;
	//const auto output_dropout_first_index = dropout.stride(0);

	const auto dropout_hidden = dropout[0];
	const auto dropout_output = dropout[1];

	const auto weights = at::cat({ weight_hh, at::cat({ weight_ch, at::zeros(weight_ch.type(), { state_size, state_size }) }, 0), weight_ih }, 1);

	const auto input_gates = gates.slice(2, state_size, 2 * state_size).contiguous();
	const auto candidate_cells = gates.slice(2, 3 * state_size).contiguous();
	const auto forget_gates = gates.slice(2, 0, state_size).contiguous();

	auto grad_X = at::zeros(X.type(), { batch_size, X_size }); // timestep specific
	auto grad_input = at::zeros(X.type(), { sequence_length, batch_size, input_size }); // across all timesteps
	//auto d_weights = at::zeros_like(weights);
	//auto d_bias = at::zeros(weights.type(), { weights.size(0) });

	const dim3 threads_1(16, 2, 2);
	const dim3 blocks_1(((gate_size + state_size) + threads_1.x - 1) / threads_1.x,
						(batch_size + threads_1.y - 1) / threads_1.y,
						(sequence_length + threads_1.z - 1) / threads_1.z);
	const dim3 threads_2(32, 2);
	const dim3 blocks_2((gate_size + threads_2.x - 1) / threads_2.x,
						(batch_size + threads_2.y - 1) / threads_2.y);
	const dim3 threads_3(32, 2);
	const dim3 blocks_3((X_size + threads_3.x - 1) / threads_3.x,
						(batch_size + threads_3.y - 1) / threads_3.y);

	AT_DISPATCH_FLOATING_TYPES(X.type(), "peephole_lstm_cuda_backward", ([&] {
		process_dgates_dtanh_dout<scalar_t> <<<blocks_1, threads_1>>> (
			X.data<scalar_t>(),
			dropout_output.data<scalar_t>(),
			input_gates.data<scalar_t>(),
			candidate_cells.data<scalar_t>(),
			gates.data<scalar_t>(),
			tanh_new_cells.data<scalar_t>(),
			grad_output.data<scalar_t>(),
			batch_x_X,
			batch_x_gate,
			batch_x_state,
			batch_size,
			X_size,
			state_size,
			gate_size,
			sequence_length);

		for (int i = sequence_length - 1; i >= 0; i--)
		{
			peephole_lstm_cuda_backward_kernel<scalar_t> <<<blocks_2, threads_2>>> (
				forget_gates.data<scalar_t>(),
				grad_output.data<scalar_t>(),
				grad_cell.clone().data<scalar_t>(),
				grad_h.data<scalar_t>(),
				grad_cell.data<scalar_t>(),
				tanh_new_cells.data<scalar_t>(),
				gates.data<scalar_t>(),
				batch_x_gate,
				batch_x_state,
				i,
				batch_size,
				state_size,
				gate_size);

			grad_X = gates[i].mm(weights);
			
			grad_in_h_c_update<scalar_t> <<<blocks_3, threads_3>>> (
				grad_X.data<scalar_t>(),
				dropout_hidden.data<scalar_t>(),
				grad_h.data<scalar_t>(),
				grad_cell.data<scalar_t>(),
				grad_input.data<scalar_t>(),
				batch_x_state,
				batch_x_input,
				i,
				input_size,
				batch_size,
				state_size,
				X_size);
			//grad_h = grad_X.slice(/*dim=*/1, 0, state_size) * dropout[0][i];
			//grad_cell += grad_X.slice(/*dim=*/1, state_size, 2 * state_size);
			//grad_input[i] = grad_X.slice(/*dim=*/1, 2 * state_size);
		}
	}));
	auto d_weights = at::mm(gates.flatten(0, 1).t(), X.flatten(0, 1));
	auto d_bias = gates.sum({ 0, 1 }, false);

	return { grad_h, grad_cell, grad_input, d_weights.slice(1, 2 * state_size), d_weights.slice(1, 0, state_size), d_weights.slice(1, state_size, 2 * state_size).slice(0, 0, 3 * state_size), d_bias };
}
