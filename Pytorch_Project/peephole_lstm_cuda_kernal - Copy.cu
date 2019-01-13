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
	const size_t gate_size,
	const size_t sig_size)
{
	const int column = blockIdx.x * blockDim.x + threadIdx.x;
	if (column < gate_size)
	{
		const int index = gates_first_idx + blockIdx.y * gate_size + column;

		if (column < sig_size) { gates[index] = sigmoid(gates[index]); }
		else { gates[index] = tanh(gates[index]); }
	}
}

template <typename scalar_t>
__global__ void peephole_lstm_cuda_forward_kernel(
	const scalar_t* __restrict__ gates,
	scalar_t* __restrict__ hc,
	scalar_t* __restrict__ output,
	scalar_t* __restrict__ tanh_new_cells,
	const size_t gates_first_idx,
	const size_t tahn_new_cells_and_output_first_idx,
	const size_t gate_size,
	const size_t state_size)
{
	const int column = blockIdx.x * blockDim.x + threadIdx.x;
	if (column < state_size)
	{
		const int gates_forget_index = gates_first_idx + blockIdx.y * gate_size + column;
		const int gates_input_index = gates_forget_index + state_size;
		const int gates_output_index = gates_input_index + state_size;
		const int gates_candidate_index = gates_output_index + state_size;

		const int tanh_new_cells_and_output_index = tahn_new_cells_and_output_first_idx + blockIdx.y * state_size + column;
		const int h_index = blockIdx.y * 2 * state_size + column;
		const int c_index = h_index + state_size;

		hc[c_index] = hc[c_index] * gates[gates_forget_index] + gates[gates_candidate_index] * gates[gates_input_index];
		tanh_new_cells[tanh_new_cells_and_output_index] = tanh(hc[c_index]);
		hc[h_index] = tanh_new_cells[tanh_new_cells_and_output_index] * gates[gates_output_index];
		output[tanh_new_cells_and_output_index] = hc[h_index];
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
		const int sequence = blockIdx.z * blockDim.z + threadIdx.z;
		if (sequence < total_sequence_length)
		{
			const int batch = blockIdx.y;
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
		const int d_gates_index = batch_x_gate * sequence_index + gate_size * blockIdx.y + column;
		if (column < state_size)
		{
			const int state_index = batch_x_state * sequence_index + state_size * blockIdx.y + column;
			const int hidden_index = state_size * blockIdx.y + column;
			d_gates[d_gates_index] *= d_tanh_new_cells[state_index] * (d_h[hidden_index] + d_outputs[state_index]) + d_cell_copy[hidden_index];
		}
		else { if (column < 2 * state_size)
		{
			const int state_index = batch_x_state * sequence_index + state_size * blockIdx.y + column - state_size;
			const int hidden_index = state_size * blockIdx.y + column - state_size;
			const auto d_new_cell = d_tanh_new_cells[state_index] * (d_h[hidden_index] + d_outputs[state_index]) + d_cell_copy[hidden_index];
			d_gates[d_gates_index] *= d_new_cell;
			d_cell[hidden_index] = forget_gates[state_index] * d_new_cell;
		}
		else { if (column < 3 * state_size)
		{
			const int state_index = batch_x_state * sequence_index + state_size * blockIdx.y + column - 2 * state_size;
			const int hidden_index = state_size * blockIdx.y + column - 2 * state_size;
			d_gates[d_gates_index] *= d_h[hidden_index] + d_outputs[state_index];
		}
		else //if (column < 4 * state_size)
		{
			const int state_index = batch_x_state * sequence_index + state_size * blockIdx.y + column - 3 * state_size;
			const int hidden_index = state_size * blockIdx.y + column - 3 * state_size;
			d_gates[d_gates_index] *= d_tanh_new_cells[state_index] * (d_h[hidden_index] + d_outputs[state_index]) + d_cell_copy[hidden_index];
		}}}
	}
}

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
	int64_t const &state_size)
{
	at::Tensor output = at::empty(weight_ih.type(), { sequence_length, batch_size, state_size });
	at::Tensor tanh_new_cells = at::empty(old_cell.type(), { sequence_length, batch_size, state_size });
	at::Tensor gates = at::matmul(input, weight_ih.transpose(0, 1));
	at::Tensor X = at::empty(weight_ih.type(), { sequence_length, batch_size, (input_size + (2 * state_size)) });
	X.slice(2, 2 * state_size) = input;

	const auto hc_size = 2 * state_size;
	const auto sig_size = 3 * state_size;
	const auto gate_size = 4 * state_size;
	const auto batch_x_gate = batch_size * gate_size;
	const auto batch_x_state = batch_size * state_size;

	const auto weight_hc_h_t = at::cat({ weight_hh, weight_ch }, 1).transpose(0, 1);
	auto hc = at::cat({ old_h, old_cell }, 1);
	
	const int threads = 64;
	const dim3 blocks_gates((gate_size + threads - 1) / threads, batch_size);
	const dim3 blocks_update((state_size + threads - 1) / threads, batch_size);

	size_t gates_first_idx = 0;
	size_t tanh_new_cells_and_output_first_idx = 0;

	AT_DISPATCH_FLOATING_TYPES(gates.type(), "peephole_lstm_cuda_forward", ([&] {
		for (int i = 0; i < sequence_length; i++)
		{
			X[i].slice(1, 0, hc_size) = hc;
			gates[i] += at::addmm(bias, hc, weight_hc_h_t);
			
			gates_activation<scalar_t> <<<blocks_gates, threads>>> (
				gates.data<scalar_t>(),
				gates_first_idx,
				gate_size,
				sig_size);
				
			peephole_lstm_cuda_forward_kernel<scalar_t> <<<blocks_update, threads>>> (
				gates.data<scalar_t>(),
				hc.data<scalar_t>(),
				output.data<scalar_t>(),
				tanh_new_cells.data<scalar_t>(),
				gates_first_idx,
				tanh_new_cells_and_output_first_idx,
				gate_size,
				state_size);
			
			gates_first_idx += batch_x_gate;
			tanh_new_cells_and_output_first_idx += batch_x_state;
		}
	}));

	at::Tensor dropout;
	if (dropout_p <= 0. || !training) { dropout = at::ones_like(output); }
	else
	{
		if (dropout_p >= 1.) { dropout = at::zeros_like(output); output *= 0; }
		else { dropout = at::bernoulli(at::empty_like(output), (1 - dropout_p)).div(1 - dropout_p); output *= dropout; }
	}

	return { output, hc.slice(1, 0, state_size), hc.slice(1, state_size), tanh_new_cells, dropout, gates, X, at::cat({ weight_hh, weight_ch, weight_ih }, /*dim=*/1) };
}

std::vector<at::Tensor> peephole_lstm_cuda_backward(
	at::Tensor &grad_output,
	at::Tensor &grad_h,
	at::Tensor &grad_cell,
	at::Tensor &tanh_new_cells,
	at::Tensor const &dropout,
	at::Tensor &gates,
	at::Tensor const &X,
	at::Tensor const &weights)
{
	const auto sequence_length = X.size(0);
	const auto batch_size = X.size(1);
	const auto state_size = grad_h.size(1);
	const auto input_size = X.size(2) - (2 * state_size);

	const auto X_size = input_size + 2 * state_size;
	const auto gate_size = 4 * state_size;
	const auto batch_x_input = batch_size * input_size;
	const auto batch_x_state = batch_size * state_size;
	const auto batch_x_X = batch_x_input + 2 * batch_x_state;
	const auto batch_x_gate = batch_x_state * 4;

	const auto input_gates = gates.slice(2, state_size, 2 * state_size).contiguous();
	const auto candidate_cells = gates.slice(2, 3 * state_size).contiguous();
	const auto forget_gates = gates.slice(2, 0, state_size).contiguous();

	auto d_X = at::zeros(X.type(), { batch_size, X_size }); // timestep specific
	auto d_input = at::zeros(X.type(), { sequence_length, batch_size, input_size }); // across all timesteps
	//auto d_weights = at::zeros_like(weights);
	//auto d_bias = at::zeros(weights.type(), { weights.size(0) });

	const dim3 threads_1(32, 1U, 32);
	const dim3 blocks_1(((gate_size + state_size) + threads_1.x - 1) / threads_1.x,
						batch_size,
						(sequence_length + threads_1.z - 1) / threads_1.z);
	const int threads_2 = 64;
	const dim3 blocks_2((gate_size + threads_2 - 1) / threads_2,
						batch_size);
	const dim3 threads_3(32, 1U, 32);
	const dim3 blocks_3((batch_x_gate + threads_3.x - 1) / threads_3.x,
						X_size,
						(sequence_length + threads_3.z) / threads_3.z);

	//const int compute_length = 50;
	AT_DISPATCH_FLOATING_TYPES(X.type(), "peephole_lstm_cuda_backward", ([&] {
		process_dgates_dtanh_dout<scalar_t> << <blocks_1, threads_1 >> > (
			X.data<scalar_t>(),
			dropout.data<scalar_t>(),
			input_gates.data<scalar_t>(),
			candidate_cells.data<scalar_t>(),
			gates.data<scalar_t>(),
			tanh_new_cells.data<scalar_t>(),
			grad_output.data<scalar_t>(),
			batch_x_X,
			batch_x_gate,
			batch_x_state,
			X_size,
			state_size,
			gate_size,
			sequence_length);

		for (int i = sequence_length - 1; i >= 0; i--)
		{
			peephole_lstm_cuda_backward_kernel<scalar_t> << <blocks_2, threads_2 >> > (
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
				state_size,
				gate_size);

			d_X = gates[i].mm(weights);
			grad_h = d_X.slice(/*dim=*/1, 0, state_size).contiguous();
			grad_cell += d_X.slice(/*dim=*/1, state_size, 2 * state_size);
			d_input[i] = d_X.slice(/*dim=*/1, 2 * state_size);

			//if (i % compute_length == 0)
			//{
			//	d_weights += at::matmul(gates.slice(0, i, i + compute_length).transpose(1, 2), X.slice(0, i, i + compute_length)).sum(/*dim=*/0, /*keepdim=*/false);
			//	d_bias += gates.slice(0, i, i + compute_length).sum(/*dims=*/{ 0, 1 }, /*keepdim=*/false);
			//}
		}
	}));
	auto d_weights = at::mm(gates.flatten(0, 1).t(), X.flatten(0, 1));
	auto d_bias = gates.sum({ 0, 1 }, false);

	return { grad_h, grad_cell, d_input, d_weights.slice(1, 2 * state_size), d_weights.slice(1, 0, state_size), d_weights.slice(1, state_size, 2 * state_size), d_bias };
}
