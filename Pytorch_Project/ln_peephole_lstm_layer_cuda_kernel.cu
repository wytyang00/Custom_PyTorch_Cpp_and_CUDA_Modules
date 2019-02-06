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
	const auto exp_n2z = exp(-2 * z);
	return (1.0 - exp_n2z) / (1.0 + exp_n2z);
}
/*
template <typename scalar_t>
__device__ __forceinline__ scalar_t d_sigmoid(scalar_t const &z)
{
	const auto exp_nz = exp(-z);
	return exp_nz / pow((1.0 + exp_nz), 2);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_tanh(scalar_t const &z)
{
	const auto exp_n2z = exp(-2 * z);
	return 4 * exp_n2z / pow((1.0 + exp_n2z), 2);
}
*/
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
__global__ void forward_part_0(
	const scalar_t* __restrict__ hidden,
	const scalar_t* __restrict__ cell,
	scalar_t* __restrict__ hiddens_storage,
	scalar_t* __restrict__ cells_storage,
	scalar_t* __restrict__ current_gate,
	const scalar_t* __restrict__ mean_fig,
	const scalar_t* __restrict__ var_fig,
	const scalar_t epsilon,
	scalar_t* __restrict__ stds_storage,
	scalar_t* __restrict__ normalized_storage,
	const scalar_t* __restrict__ gamma_fig,
	const scalar_t* __restrict__ bias_fig,
	scalar_t* __restrict__ activated_storage,
	scalar_t* __restrict__ forgotten_cell,
	const scalar_t* __restrict__ dropout_candidate_cell,
	const int64_t batch_size,
	const int64_t state_size,
	const int64_t state_size_3)
{
	const int column = blockIdx.x * blockDim.x + threadIdx.x;
	if (column < state_size)
	{
		const int batch = blockIdx.y * blockDim.y + threadIdx.y;
		if (batch < batch_size)
		{
			const int process_idx = blockIdx.z;
			if (process_idx < 4)
			{
				if (process_idx < 3) //Normalizations and stuff
				{
					const int mean_var_std_idx = batch * 3 + process_idx;
					const scalar_t std = sqrt(var_fig[mean_var_std_idx] + epsilon);
					if (column == 0)
					{
						stds_storage[mean_var_std_idx] = std;
					}
					const int gate_val_storage_idx = batch * state_size_3 + process_idx * state_size + column;
					const int gate_val_local_idx = gate_val_storage_idx + batch * state_size;
					scalar_t gate_val = (current_gate[gate_val_local_idx] - mean_fig[mean_var_std_idx]) / std;
					normalized_storage[gate_val_storage_idx] = gate_val;
					const int gamma_bias_idx = process_idx * state_size + column;
					if (process_idx < 2) //forget gate & input gate
					{
						gate_val = sigmoid((gate_val * gamma_fig[gamma_bias_idx]) + bias_fig[gamma_bias_idx]);
						if (process_idx == 0) //forget cell memory
						{
							const int local_state_idx = batch * state_size + column;
							forgotten_cell[local_state_idx] = gate_val * cell[local_state_idx];
						}
						activated_storage[gate_val_storage_idx] = gate_val;
						current_gate[gate_val_local_idx] = gate_val;
					}
					else //candidate cell
					{
						gate_val = tanh((gate_val * gamma_fig[gamma_bias_idx]) + bias_fig[gamma_bias_idx]);
						activated_storage[gate_val_storage_idx] = gate_val;
						current_gate[gate_val_local_idx] = gate_val * dropout_candidate_cell[batch * state_size + column];
					}
				}
				else //Hidden, Cell Storage
				{
					const int local_state_idx = batch * state_size + column;
					hiddens_storage[local_state_idx] = hidden[local_state_idx];
					cells_storage[local_state_idx] = cell[local_state_idx];
				}
			}
		}
	}
}

template <typename scalar_t>
__global__ void forward_part_1(
	const scalar_t* __restrict__ forgotten_cell,
	const scalar_t* __restrict__ current_gate,
	scalar_t* __restrict__ cell,
	const int64_t batch_size,
	const int64_t state_size,
	const int64_t gate_size)
{
	const int column = blockIdx.x * blockDim.x + threadIdx.x;
	if (column < state_size)
	{
		const int batch = blockIdx.y * blockDim.y + threadIdx.y;
		if (batch < batch_size)
		{
			const int local_state_idx = batch * state_size + column;
			const int local_input_gate_idx = batch * gate_size + state_size + column;
			cell[local_state_idx] = forgotten_cell[local_state_idx] + current_gate[local_input_gate_idx] * current_gate[local_input_gate_idx + state_size];
		}
	}
}

template <typename scalar_t>
__global__ void forward_part_2(
	scalar_t* __restrict__ cell,
	const scalar_t* __restrict__ mean,
	const scalar_t* __restrict__ var,
	const scalar_t epsilon,
	scalar_t* __restrict__ new_cell_stds_storage,
	scalar_t* __restrict__ new_cell_normalized_storage,
	const scalar_t* __restrict__ gamma_new_cell,
	const scalar_t* __restrict__ beta_new_cell,
	scalar_t* __restrict__ hc,
	scalar_t* __restrict__ current_gate,
	const scalar_t* __restrict__ weight_co,
	const int64_t batch_size,
	const int64_t state_size,
	const int64_t state_size_2)
{
	const int column = blockIdx.x * blockDim.x + threadIdx.x;
	if (column < state_size)
	{
		const int batch = blockIdx.y * blockDim.y + threadIdx.y;
		if (batch < batch_size)
		{
			const int cell_idx = batch * state_size + column;
			const scalar_t std = sqrt(var[batch] + epsilon);
			if (column == 0)
			{
				new_cell_stds_storage[batch] = std;
			}
			scalar_t cell_val = (cell[cell_idx] - mean[batch]) / std;
			new_cell_normalized_storage[cell_idx] = cell_val;
			cell_val = (cell_val * gamma_new_cell[column]) + beta_new_cell[column];
			cell[cell_idx] = cell_val;
			const int hc_idx = cell_idx + (batch + 1) * state_size;
			hc[hc_idx] = cell_val;
			current_gate[hc_idx + (batch + 1) * state_size_2] += cell_val * weight_co[column];
		}
	}
}

template <typename scalar_t>
__global__ void forward_part_3(
	const scalar_t* __restrict__ current_gate,
	const scalar_t* __restrict__ mean,
	const scalar_t* __restrict__ var,
	const scalar_t epsilon,
	scalar_t* __restrict__ output_gate_stds_storage,
	scalar_t* __restrict__ output_gate_normalized_storage,
	const scalar_t* __restrict__ gamma_o,
	const scalar_t* __restrict__ bias_o,
	scalar_t* __restrict__ output_gate_activated_storage,
	const scalar_t* __restrict__ cell,
	scalar_t* __restrict__ tanh_new_cell_storage,
	scalar_t* __restrict__ hidden,
	scalar_t* __restrict__ hc,
	scalar_t* __restrict__ outputs,
	const scalar_t* __restrict__ dropout_output,
	const int64_t batch_size,
	const int64_t state_size,
	const int64_t state_size_3)
{
	const int column = blockIdx.x * blockDim.x + threadIdx.x;
	if (column < state_size)
	{
		const int batch = blockIdx.y * blockDim.y + threadIdx.y;
		if (batch < batch_size)
		{
			const int state_and_output_gate_storage_idx = batch * state_size + column;
			const int hc_idx = state_and_output_gate_storage_idx + batch * state_size;
			const int output_gate_idx = state_and_output_gate_storage_idx + (batch + 1) * state_size_3;
			const scalar_t std = sqrt(var[batch] + epsilon);
			if (column == 0)
			{
				output_gate_stds_storage[batch] = std;
			}
			scalar_t output_gate_val = (current_gate[output_gate_idx] - mean[batch]) / std;
			output_gate_normalized_storage[state_and_output_gate_storage_idx] = output_gate_val;
			output_gate_val = sigmoid((output_gate_val * gamma_o[column]) + bias_o[column]);
			output_gate_activated_storage[state_and_output_gate_storage_idx] = output_gate_val;
			const scalar_t tanh_cell = tanh(cell[state_and_output_gate_storage_idx]);
			tanh_new_cell_storage[state_and_output_gate_storage_idx] = tanh_cell;
			const scalar_t hidden_val = output_gate_val * tanh_cell;
			hidden[state_and_output_gate_storage_idx] = hidden_val;
			hc[hc_idx] = hidden_val;
			outputs[state_and_output_gate_storage_idx] = hidden_val * dropout_output[state_and_output_gate_storage_idx];
		}
	}
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
	int64_t const &gate_size)
{
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
	auto forgotten_cell = at::empty_like(cell);
	at::Tensor mean;
	at::Tensor var;

	const dim3 threads(32, 8);
	const dim3 blocks_0((state_size + threads.x - 1) / threads.x,
		                (batch_size + threads.y - 1) / threads.y,
						4);
	const dim3 blocks_1((state_size + threads.x - 1) / threads.x,
		                (batch_size + threads.y - 1) / threads.y);

	AT_DISPATCH_FLOATING_TYPES(ih.type(), "ln_peephole_lstm_layer_cuda_forward", ([&] {
		for (int i = 0; i < sequence_length; i++)
		{
			current_gate = at::addmm(ih[i], hc, weight_hc_h).view({ batch_size, 4, state_size });
			mean = current_gate.slice(1, 0, 3).mean(/*dim=*/2, /*keepdim=*/false);
			var = current_gate.slice(1, 0, 3).var(/*dim=*/2, /*unbiased=*/false, /*keepdim=*/false);

			forward_part_0<scalar_t> <<<blocks_0, threads>>> (
				hidden.data<scalar_t>(),
				cell.data<scalar_t>(),
				hiddens[i].data<scalar_t>(),
				cells[i].data<scalar_t>(),
				current_gate.data<scalar_t>(),
				mean.data<scalar_t>(),
				var.data<scalar_t>(),
				epsilon,
				gates_fig_stds[i].data<scalar_t>(),
				gates_fig_normalized[i].data<scalar_t>(),
				gamma_fig.data<scalar_t>(),
				bias_fig.data<scalar_t>(),
				gates_fig[i].data<scalar_t>(),
				forgotten_cell.data<scalar_t>(),
				dropout_candidate_cell[i].data<scalar_t>(),
				batch_size,
				state_size,
				state_size_3);

			forward_part_1<scalar_t> <<<blocks_1, threads>>> (
				forgotten_cell.data<scalar_t>(),
				current_gate.data<scalar_t>(),
				cell.data<scalar_t>(),
				batch_size,
				state_size,
				gate_size);

			mean = cell.mean(/*dim=*/1, /*keepdim=*/false);
			var = cell.var(/*dim=*/1, /*unbiased=*/false, /*keepdim=*/false);

			forward_part_2<scalar_t> <<<blocks_1, threads>>> (
				cell.data<scalar_t>(),
				mean.data<scalar_t>(),
				var.data<scalar_t>(),
				epsilon,
				new_cells_stds[i].data<scalar_t>(),
				new_cells_normalized[i].data<scalar_t>(),
				gamma_new_cell.data<scalar_t>(),
				beta_new_cell.data<scalar_t>(),
				hc.data<scalar_t>(),
				current_gate.data<scalar_t>(),
				weight_co.data<scalar_t>(),
				batch_size,
				state_size,
				state_size_2);

			mean = current_gate.select(1, 3).mean(/*dim=*/1, /*keepdim=*/false);
			var = current_gate.select(1, 3).var(/*dim=*/1, /*unbiased=*/false, /*keepdim=*/false);

			forward_part_3<scalar_t> <<<blocks_1, threads>>> (
				current_gate.data<scalar_t>(),
				mean.data<scalar_t>(),
				var.data<scalar_t>(),
				epsilon,
				gates_o_stds[i].data<scalar_t>(),
				gates_o_normalized[i].data<scalar_t>(),
				gamma_o.data<scalar_t>(),
				bias_o.data<scalar_t>(),
				gates_o[i].data<scalar_t>(),
				cell.data<scalar_t>(),
				tanh_new_cells[i].data<scalar_t>(),
				hidden.data<scalar_t>(),
				hc.data<scalar_t>(),
				outputs[i].data<scalar_t>(),
				dropout_output[i].data<scalar_t>(),
				batch_size,
				state_size,
				state_size_3);
		}
	}));
	cells[sequence_length] = cell;

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

////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename scalar_t>
__global__ void backward_preparation(
	scalar_t* __restrict__ grad_output,
	const scalar_t* __restrict__ dropout_output,
	const scalar_t* __restrict__ dropout_candidate_cell,
	const scalar_t* __restrict__ cells,
	const scalar_t* __restrict__ gates_fig,
	const scalar_t* __restrict__ gates_o,
	scalar_t* __restrict__ grad_gates_layer_normalized,
	scalar_t* __restrict__ gates_fig_stds,
	scalar_t* __restrict__ gates_o_stds,
	scalar_t* __restrict__ new_cells_stds,
	const scalar_t* __restrict__ tanh_new_cells,
	scalar_t* __restrict__ grad_new_cells_wrt_tanh_new_cell,
	scalar_t* __restrict__ forget_gates,
	const size_t n_total_batches,
	const size_t state_size,
	const size_t state_size_2,
	const size_t state_size_3,
	const size_t gate_size)
{
	const int column = blockIdx.x * blockDim.x + threadIdx.x;
	if (column < state_size)
	{
		const int batch = blockIdx.y * blockDim.y + threadIdx.y;
		if (batch < n_total_batches)
		{
			const int process_idx = blockIdx.z;
			if (process_idx < 6)
			{
				if (process_idx == 0)
				{
					const int cell_forget_idx = batch * state_size + column;
					const scalar_t forget_gate_val = gates_fig[batch * state_size_3 + column];
					forget_gates[cell_forget_idx] = forget_gate_val;
					grad_gates_layer_normalized[batch * gate_size + column]
						= cells[cell_forget_idx]
						* d_sigmoid_with_output(forget_gate_val);
					if (column == 0)
					{
						gates_fig_stds[batch * 3] *= state_size;
					}
				}
				else{if (process_idx == 1)
				{
					const int dropout_idx = batch * state_size + column;
					const int candidate_cell_idx = dropout_idx + (batch + 1) * state_size_2;
					const int input_gate_idx = candidate_cell_idx - state_size;
					const int store_idx = input_gate_idx + batch * state_size;
					grad_gates_layer_normalized[store_idx] = gates_fig[candidate_cell_idx]
						                                     * d_sigmoid_with_output(gates_fig[input_gate_idx])
						                                     * dropout_candidate_cell[dropout_idx];
					if (column == 0)
					{
						gates_fig_stds[batch * 3 + 1] *= state_size;
					}
				}
				else{if (process_idx == 2)
				{
					const int dropout_idx = batch * state_size + column;
					const int candidate_cell_idx = dropout_idx + (batch + 1) * state_size_2;
					const int input_gate_idx = candidate_cell_idx - state_size;
					const int store_idx = candidate_cell_idx + batch * state_size;
					grad_gates_layer_normalized[store_idx] = gates_fig[input_gate_idx]
						                                     * d_tanh_with_output(gates_fig[candidate_cell_idx])
						                                     * dropout_candidate_cell[dropout_idx];
					if (column == 0)
					{
						gates_fig_stds[batch * 3 + 2] *= state_size;
					}
				}
				else{if (process_idx == 3)
				{
					const int tanh_and_output_idx = batch * state_size + column;
					grad_gates_layer_normalized[batch * gate_size + column + state_size_3]
						= tanh_new_cells[tanh_and_output_idx]
						* d_sigmoid_with_output(gates_o[tanh_and_output_idx]);
					if (column == 0)
					{
						gates_o_stds[batch] *= state_size;
					}
				}
				else{if (process_idx == 4)
				{
					const int index = batch * state_size + column;
					grad_output[index] *= dropout_output[index];
					if (column == 0)
					{
						new_cells_stds[batch] *= state_size;
					}
				}
				else{if (process_idx == 5)
				{
					const int index = batch * state_size + column;
					grad_new_cells_wrt_tanh_new_cell[index] = d_tanh_with_output(tanh_new_cells[index]) * gates_o[index];
				}}}}}}
			}
		}
	}
}

template <typename scalar_t>
__global__ void backward_loop_part_0(
	const scalar_t* __restrict__ grad_hidden,
	scalar_t* __restrict__ grad_new_cell_wrt_tanh_new_cell,
	const scalar_t* __restrict__ grad_output,
	scalar_t* __restrict__ grad_gate_layer_normalized,
	const scalar_t* __restrict__ gamma_o,
	scalar_t* __restrict__ grad_output_gate_normalized,
	const size_t batch_size,
	const size_t state_size,
	const size_t state_size_3)
{
	const int column = blockIdx.x * blockDim.x + threadIdx.x;
	if (column < state_size)
	{
		const int batch = blockIdx.y * blockDim.y + threadIdx.y;
		if (batch < batch_size)
		{
			const int local_state_idx = batch * state_size + column;
			const int gate_idx = local_state_idx + (batch + 1) * state_size_3;
			scalar_t grad_val = grad_hidden[local_state_idx] + grad_output[local_state_idx];
			grad_new_cell_wrt_tanh_new_cell[local_state_idx] *= grad_val;
			grad_val *= grad_gate_layer_normalized[gate_idx];
			grad_gate_layer_normalized[gate_idx] = grad_val;
			grad_val *= gamma_o[column];
			grad_output_gate_normalized[local_state_idx] = grad_val;
		}
	}
}

template <typename scalar_t>
__global__ void backward_loop_part_1(
	const scalar_t* __restrict__ grad_output_gate_normalized,
	const scalar_t* __restrict__ grad_output_gate_normalized_sum,
	const scalar_t* __restrict__ grad_output_gate_normalized_prod_sum,
	const scalar_t* __restrict__ output_gate_normalized,
	const scalar_t* __restrict__ output_gate_std,
	scalar_t* __restrict__ grad_gate_raw,
	const scalar_t* __restrict__ weight_co,
	const scalar_t* __restrict__ grad_new_cell_wrt_tanh_new_cell,
	scalar_t* __restrict__ grad_cell,
	scalar_t* __restrict__ grad_new_cell,
	const scalar_t* __restrict__ gamma_new_cell,
	const size_t batch_size,
	const size_t state_size,
	const size_t state_size_3)
{
	const int column = blockIdx.x * blockDim.x + threadIdx.x;
	if (column < state_size)
	{
		const int batch = blockIdx.y * blockDim.y + threadIdx.y;
		if (batch < batch_size)
		{
			const int local_state_idx = batch * state_size + column;
			const int gate_idx = local_state_idx + (batch + 1) * state_size_3;
			scalar_t grad_val = (state_size * grad_output_gate_normalized[local_state_idx]
								 - grad_output_gate_normalized_sum[batch]
								 - output_gate_normalized[local_state_idx] * grad_output_gate_normalized_prod_sum[batch]) / output_gate_std[batch];
			grad_gate_raw[gate_idx] = grad_val;
			grad_val = grad_val * weight_co[column] + grad_new_cell_wrt_tanh_new_cell[local_state_idx] + grad_cell[local_state_idx];
			grad_new_cell[local_state_idx] = grad_val;
			grad_cell[local_state_idx] = grad_val * gamma_new_cell[column];
		}
	}
}

template <typename scalar_t>
__global__ void backward_loop_part_2(
	scalar_t* __restrict__ grad_cell,
	const scalar_t* __restrict__ grad_cell_sum,
	const scalar_t* __restrict__ grad_cell_prod_sum,
	const scalar_t* __restrict__ new_cell_normalized,
	const scalar_t* __restrict__ new_cell_std,
	const size_t batch_size,
	const size_t state_size)
{
	const int column = blockIdx.x * blockDim.x + threadIdx.x;
	if (column < state_size)
	{
		const int batch = blockIdx.y * blockDim.y + threadIdx.y;
		if (batch < batch_size)
		{
			const int local_state_idx = batch * state_size + column;
			grad_cell[local_state_idx] = (state_size * grad_cell[local_state_idx]
									      - grad_cell_sum[batch]
								          - new_cell_normalized[local_state_idx] * grad_cell_prod_sum[batch]) / new_cell_std[batch];
		}
	}
}

template <typename scalar_t>
__global__ void backward_loop_part_3(
	const scalar_t* __restrict__ grad_cell,
	scalar_t* __restrict__ grad_gate_layer_normalized,
	const scalar_t* __restrict__ gamma_f,
	const scalar_t* __restrict__ gamma_i,
	const scalar_t* __restrict__ gamma_g,
	scalar_t* __restrict__ grad_fig_gate_normalized,
	const size_t batch_size,
	const size_t state_size,
	const size_t state_size_3)
{
	const int column = blockIdx.x * blockDim.x + threadIdx.x;
	if (column < state_size)
	{
		const int batch = blockIdx.y * blockDim.y + threadIdx.y;
		if (batch < batch_size)
		{
			const int process_idx = blockIdx.z;
			if (process_idx < 3)
			{
				const int local_state_idx = batch * state_size + column;
				const int fig_idx = batch * state_size_3 + process_idx * state_size + column;
				const int gate_idx = fig_idx + batch * state_size;
				scalar_t grad_val = grad_cell[local_state_idx] * grad_gate_layer_normalized[gate_idx];
				grad_gate_layer_normalized[gate_idx] = grad_val;
				if (process_idx == 0)
				{
					grad_val *= gamma_f[column];
				}
				else{if (process_idx == 1)
				{
					grad_val *= gamma_i[column];
				}
				else
				{
					grad_val *= gamma_g[column];
				}}
				grad_fig_gate_normalized[fig_idx] = grad_val;
			}
		}
	}
}

template <typename scalar_t>
__global__ void backward_loop_part_4(
	const scalar_t* __restrict__ grad_fig_gate_normalized,
	const scalar_t* __restrict__ grad_fig_gate_normalized_sum,
	const scalar_t* __restrict__ grad_fig_gate_normalized_prod_sum,
	const scalar_t* __restrict__ gate_fig_normalized,
	const scalar_t* __restrict__ gate_fig_std,
	scalar_t* __restrict__ grad_gate_raw,
	const size_t batch_size,
	const size_t state_size,
	const size_t state_size_3)
{
	const int column = blockIdx.x * blockDim.x + threadIdx.x;
	if (column < state_size)
	{
		const int batch = blockIdx.y * blockDim.y + threadIdx.y;
		if (batch < batch_size)
		{
			const int process_idx = blockIdx.z;
			if (process_idx < 3)
			{
				const int fig_idx = batch * state_size_3 + process_idx * state_size + column;
				const int reduced_fig_idx = batch * 3 + process_idx;
				scalar_t grad_val = (state_size * grad_fig_gate_normalized[fig_idx]
									 - grad_fig_gate_normalized_sum[reduced_fig_idx]
									 - gate_fig_normalized[fig_idx] * grad_fig_gate_normalized_prod_sum[reduced_fig_idx]) / gate_fig_std[reduced_fig_idx];
				grad_gate_raw[fig_idx + batch * state_size] = grad_val;
			}
		}
	}
}

template <typename scalar_t>
__global__ void backward_loop_part_5(
	const scalar_t* __restrict__ grad_hci,
	const scalar_t* __restrict__ forget_gate,
	scalar_t* __restrict__ grad_hidden,
	scalar_t* __restrict__ grad_cell,
	scalar_t* __restrict__ grad_input,
	const size_t batch_size,
	const size_t input_size,
	const size_t state_size,
	const size_t state_size_2,
	const size_t X_size)
{
	const int column = blockIdx.x * blockDim.x + threadIdx.x;
	if (column < X_size)
	{
		const int batch = blockIdx.y * blockDim.y + threadIdx.y;
		if (batch < batch_size)
		{
			const int grad_idx = batch * X_size + column;
			if (column < state_size)
			{
				grad_hidden[batch * state_size + column] = grad_hci[grad_idx];
			}
			else{if (column < state_size_2)
			{
				const int local_state_idx = (batch - 1) * state_size + column;
				grad_cell[local_state_idx] = grad_hci[grad_idx] + grad_cell[local_state_idx] * forget_gate[local_state_idx];
			}
			else
			{
				grad_input[batch * input_size + column - state_size_2] = grad_hci[grad_idx];
			}}
		}
	}
}

template <typename scalar_t>
__global__ void backward_final(
	scalar_t* __restrict__ sum_to_get_grads,
	const scalar_t* __restrict__ grad_gates_raw,
	const scalar_t* __restrict__ cells,
	const scalar_t* __restrict__ grad_gates_layer_normalized,
	const scalar_t* __restrict__ gates_fig_normalized,
	const scalar_t* __restrict__ gates_o_normalized,
	const scalar_t* __restrict__ grad_new_cells,
	const scalar_t* __restrict__ new_cells_normalized,
	const size_t n_total_batches,
	const size_t batch_size,
	const size_t state_size,
	const size_t state_size_2,
	const size_t state_size_3,
	const size_t gate_size)
{
	const int column = blockIdx.x * blockDim.x + threadIdx.x;
	if (column < state_size)
	{
		const int batch = blockIdx.y * blockDim.y + threadIdx.y;
		if (batch < n_total_batches)
		{
			const int process_idx = blockIdx.z;
			if (process_idx < 13)
			{
				const int store_idx = batch * (gate_size * 3 + state_size) + process_idx * state_size + column;
				/*if (process_idx == 0)
				{
					sum_to_get_grads[store_idx] = grad_gates_raw[batch * gate_size + column] * cells[batch * state_size + column];
				}
				else{
				if (process_idx == 1)
				{
					sum_to_get_grads[store_idx] = grad_gates_raw[batch * gate_size + column + state_size] * cells[batch * state_size + column];
				}
				else{
				if (process_idx == 2)
				{
					sum_to_get_grads[store_idx] = grad_gates_raw[batch * gate_size + column + state_size_3] * cells[(batch + batch_size) * state_size + column];
				}
				else{
				if (process_idx == 3)
				{
					sum_to_get_grads[store_idx] = grad_gates_layer_normalized[batch * gate_size + column];
				}
				else{
				if (process_idx == 4)
				{
					sum_to_get_grads[store_idx] = grad_gates_layer_normalized[batch * gate_size + column + state_size];
				}
				else{
				if (process_idx == 5)
				{
					sum_to_get_grads[store_idx] = grad_gates_layer_normalized[batch * gate_size + column + state_size_2];
				}
				else{
				if (process_idx == 6)
				{
					sum_to_get_grads[store_idx] = grad_gates_layer_normalized[batch * gate_size + column + state_size_3];
				}
				else{
				if (process_idx == 7)
				{
					sum_to_get_grads[store_idx] = grad_gates_layer_normalized[batch * gate_size + column] * gates_fig_normalized[batch * state_size_3 + column];
				}
				else{
				if (process_idx == 8)
				{
					const int norm_idx = batch * state_size_3 + column + state_size;
					sum_to_get_grads[store_idx] = grad_gates_layer_normalized[norm_idx + batch * state_size] * gates_fig_normalized[norm_idx];
				}
				else{
				if (process_idx == 9)
				{
					const int norm_idx = batch * state_size_3 + column + state_size_2;
					sum_to_get_grads[store_idx] = grad_gates_layer_normalized[norm_idx + batch * state_size] * gates_fig_normalized[norm_idx];
				}
				else{
				if (process_idx == 10)
				{
					const int norm_idx = batch * state_size + column;
					sum_to_get_grads[store_idx] = grad_gates_layer_normalized[norm_idx + (batch + 1) * state_size_3] * gates_o_normalized[norm_idx];
				}
				else{
				if (process_idx == 11)
				{
					sum_to_get_grads[store_idx] = grad_new_cells[batch * state_size + column] * new_cells_normalized[batch * state_size + column];
				}
				else
				{
					sum_to_get_grads[store_idx] = grad_new_cells[batch * state_size + column];
				}}}}}}}}}}}}*/
				switch (process_idx)
				{
				case 0:
				{
					sum_to_get_grads[store_idx] = grad_gates_raw[batch * gate_size + column] * cells[batch * state_size + column];
					break;
				}
				case 1:
				{
					sum_to_get_grads[store_idx] = grad_gates_raw[batch * gate_size + column + state_size] * cells[batch * state_size + column];
					break;
				}
				case 2:
				{
					sum_to_get_grads[store_idx] = grad_gates_raw[batch * gate_size + column + state_size_3] * cells[(batch + batch_size) * state_size + column];
					break;
				}
				case 3:
				{
					sum_to_get_grads[store_idx] = grad_gates_layer_normalized[batch * gate_size + column];
					break;
				}
				case 4:
				{
					sum_to_get_grads[store_idx] = grad_gates_layer_normalized[batch * gate_size + column + state_size];
					break;
				}
				case 5:
				{
					sum_to_get_grads[store_idx] = grad_gates_layer_normalized[batch * gate_size + column + state_size_2];
					break;
				}
				case 6:
				{
					sum_to_get_grads[store_idx] = grad_gates_layer_normalized[batch * gate_size + column + state_size_3];
					break;
				}
				case 7:
				{
					sum_to_get_grads[store_idx] = grad_gates_layer_normalized[batch * gate_size + column] * gates_fig_normalized[batch * state_size_3 + column];
					break;
				}
				case 8:
				{
					const int norm_idx = batch * state_size_3 + column + state_size;
					sum_to_get_grads[store_idx] = grad_gates_layer_normalized[norm_idx + batch * state_size] * gates_fig_normalized[norm_idx];
					break;
				}
				case 9:
				{
					const int norm_idx = batch * state_size_3 + column + state_size_2;
					sum_to_get_grads[store_idx] = grad_gates_layer_normalized[norm_idx + batch * state_size] * gates_fig_normalized[norm_idx];
					break;
				}
				case 10:
				{
					const int norm_idx = batch * state_size + column;
					sum_to_get_grads[store_idx] = grad_gates_layer_normalized[norm_idx + (batch + 1) * state_size_3] * gates_o_normalized[norm_idx];
					break;
				}
				case 11:
				{
					sum_to_get_grads[store_idx] = grad_new_cells[batch * state_size + column] * new_cells_normalized[batch * state_size + column];
					break;
				}
				case 12:
				{
					sum_to_get_grads[store_idx] = grad_new_cells[batch * state_size + column];
					break;
				}
				}
			}
		}
	}
}

std::vector<at::Tensor> ln_peephole_lstm_layer_cuda_backward(
	at::Tensor &grad_output,
	at::Tensor &grad_hidden,
	at::Tensor &grad_cell,
	at::Tensor const &input,
	at::Tensor const &hiddens,
	at::Tensor const &cells,
	at::Tensor const &gates_fig,
	at::Tensor const &gates_fig_normalized,
	at::Tensor &gates_fig_stds,
	at::Tensor const &gates_o,
	at::Tensor const &gates_o_normalized,
	at::Tensor &gates_o_stds,
	at::Tensor const &new_cells_normalized,
	at::Tensor &new_cells_stds,
	at::Tensor &tanh_new_cells,
	at::Tensor const &dropout,
	at::Tensor const &weight_ih,
	at::Tensor const &weight_hh,
	at::Tensor const &weight_ch,
	at::Tensor const &gamma_f,
	at::Tensor const &gamma_i,
	at::Tensor const &gamma_g,
	at::Tensor const &gamma_o,
	at::Tensor const &gamma_new_cell)
{
	const auto sequence_length = input.size(0);
	const auto batch_size = input.size(1);
	const auto state_size = hiddens.size(2);
	const auto state_size_2 = state_size + state_size;
	const auto state_size_3 = state_size_2 + state_size;
	const auto gate_size = state_size_3 + state_size;
	const auto input_size = input.size(2);
	const auto X_size = input_size + state_size_2;

	const auto n_total_batches = batch_size * sequence_length;

	const auto dropout_candidate_cell = dropout[0];
	const auto dropout_output = dropout[1];

	const auto weights = at::cat({ weight_hh,
								   at::cat({ weight_ch.slice(0, 0, state_size).diag(),
											 weight_ch.slice(0, state_size, state_size_2).diag(),
											 at::zeros({ state_size_2, state_size }, weight_ch.options()) }),
								   weight_ih }, 1);
	const auto weight_co = weight_ch.slice(0, state_size_2);

	auto grad_input = at::empty_like(input);

	auto grad_gates_layer_normalized = at::empty({ sequence_length, batch_size, gate_size }, gates_fig.options());
	auto grad_gates_raw = at::empty_like(grad_gates_layer_normalized);
	auto grad_new_cells = at::empty_like(tanh_new_cells);
	auto grad_new_cells_wrt_tanh_new_cell = at::empty_like(tanh_new_cells);

	auto grad_output_gate_normalized = at::empty({ batch_size, state_size }, grad_gates_raw.options());
	auto grad_fig_gate_normalized = at::empty({ batch_size, 3, state_size }, grad_gates_raw.options());
	auto forget_gates = at::empty_like(tanh_new_cells);

	at::Tensor sum_to_get_grads;

	const dim3 threads(64, 8);
	const dim3 blocks_0((state_size + threads.x - 1) / threads.x,
						(n_total_batches + threads.y - 1) / threads.y,
						6);
	const dim3 blocks_1((state_size + threads.x - 1) / threads.x,
						(batch_size + threads.y - 1) / threads.y);
	const dim3 blocks_2((state_size + threads.x - 1) / threads.x,
						(batch_size + threads.y - 1) / threads.y,
						3);
	const dim3 blocks_3((X_size + threads.x - 1) / threads.x,
						(batch_size + threads.y - 1) / threads.y);
	const dim3 blocks_4((state_size + threads.x - 1) / threads.x,
						(n_total_batches + threads.y - 1) / threads.y,
						13);

	AT_DISPATCH_FLOATING_TYPES(gates_fig.type(), "ln_peephole_lstm_layer_cuda_backward", ([&] {
		backward_preparation<scalar_t> <<<blocks_0, threads>>> (
			grad_output.data<scalar_t>(),
			dropout_output.data<scalar_t>(),
			dropout_candidate_cell.data<scalar_t>(),
			cells.data<scalar_t>(),
			gates_fig.data<scalar_t>(),
			gates_o.data<scalar_t>(),
			grad_gates_layer_normalized.data<scalar_t>(),
			gates_fig_stds.data<scalar_t>(),
			gates_o_stds.data<scalar_t>(),
			new_cells_stds.data<scalar_t>(),
			tanh_new_cells.data<scalar_t>(),
			grad_new_cells_wrt_tanh_new_cell.data<scalar_t>(),
			forget_gates.data<scalar_t>(),
			n_total_batches,
			state_size,
			state_size_2,
			state_size_3,
			gate_size);

		for (int i = sequence_length - 1; i >= 0; i--)
		{
			backward_loop_part_0<scalar_t> <<<blocks_1, threads>>> (
				grad_hidden.data<scalar_t>(),
				grad_new_cells_wrt_tanh_new_cell[i].data<scalar_t>(),
				grad_output[i].data<scalar_t>(),
				grad_gates_layer_normalized[i].data<scalar_t>(),
				gamma_o.data<scalar_t>(),
				grad_output_gate_normalized.data<scalar_t>(),
				batch_size,
				state_size,
				state_size_3);

			backward_loop_part_1<scalar_t> <<<blocks_1, threads>>> (
				grad_output_gate_normalized.data<scalar_t>(),
				grad_output_gate_normalized.sum(/*dim=*/1, /*keepdim=*/false).data<scalar_t>(),
				grad_output_gate_normalized.mul(gates_o_normalized[i]).sum(/*dim=*/1, /*keepdim=*/false).data<scalar_t>(),
				gates_o_normalized[i].data<scalar_t>(),
				gates_o_stds[i].data<scalar_t>(),
				grad_gates_raw[i].data<scalar_t>(),
				weight_co.data<scalar_t>(),
				grad_new_cells_wrt_tanh_new_cell[i].data<scalar_t>(),
				grad_cell.data<scalar_t>(),
				grad_new_cells[i].data<scalar_t>(),
				gamma_new_cell.data<scalar_t>(),
				batch_size,
				state_size,
				state_size_3);

			backward_loop_part_2<scalar_t> <<<blocks_1, threads>>> (
				grad_cell.data<scalar_t>(),
				grad_cell.sum(/*dim=*/1, /*keepdim=*/false).data<scalar_t>(),
				grad_cell.mul(new_cells_normalized[i]).sum(/*dim=*/1, /*keepdim=*/false).data<scalar_t>(),
				new_cells_normalized[i].data<scalar_t>(),
				new_cells_stds[i].data<scalar_t>(),
				batch_size,
				state_size);

			backward_loop_part_3<scalar_t> <<<blocks_2, threads>>> (
				grad_cell.data<scalar_t>(),
				grad_gates_layer_normalized[i].data<scalar_t>(),
				gamma_f.data<scalar_t>(),
				gamma_i.data<scalar_t>(),
				gamma_g.data<scalar_t>(),
				grad_fig_gate_normalized.data<scalar_t>(),
				batch_size,
				state_size,
				state_size_3);

			backward_loop_part_4<scalar_t> <<<blocks_2, threads>>> (
				grad_fig_gate_normalized.data<scalar_t>(),
				grad_fig_gate_normalized.sum(/*dim=*/2, /*keepdim=*/false).data<scalar_t>(),
				grad_fig_gate_normalized.mul(gates_fig_normalized[i]).sum(/*dim=*/2, /*keepdim=*/false).data<scalar_t>(),
				gates_fig_normalized[i].data<scalar_t>(),
				gates_fig_stds[i].data<scalar_t>(),
				grad_gates_raw[i].data<scalar_t>(),
				batch_size,
				state_size,
				state_size_3);

			backward_loop_part_5<scalar_t> <<<blocks_3, threads>>> (
				grad_gates_raw[i].mm(weights).data<scalar_t>(),
				forget_gates[i].data<scalar_t>(),
				grad_hidden.data<scalar_t>(),
				grad_cell.data<scalar_t>(),
				grad_input[i].data<scalar_t>(),
				batch_size,
				input_size,
				state_size,
				state_size_2,
				X_size);
		}
		sum_to_get_grads = at::empty({ sequence_length * batch_size, gate_size * 3 + state_size }, weights.options());
		backward_final<scalar_t> <<<blocks_4, threads>>> (
			sum_to_get_grads.data<scalar_t>(),
			grad_gates_raw.data<scalar_t>(),
			cells.data<scalar_t>(),
			grad_gates_layer_normalized.data<scalar_t>(),
			gates_fig_normalized.data<scalar_t>(),
			gates_o_normalized.data<scalar_t>(),
			grad_new_cells.data<scalar_t>(),
			new_cells_normalized.data<scalar_t>(),
			n_total_batches,
			batch_size,
			state_size,
			state_size_2,
			state_size_3,
			gate_size);
	}));
	const auto grad_weight_ih_hh = grad_gates_raw.view({ sequence_length * batch_size, gate_size }).t().mm(at::cat({ input, hiddens }, 2).view({ sequence_length * batch_size, input_size + state_size }));

	const auto bunch_of_grads = sum_to_get_grads.sum(/*dim=*/0, /*keepdim=*/false);

	return { grad_input,
			 grad_weight_ih_hh.slice(1, 0, input_size).contiguous(),
			 grad_weight_ih_hh.slice(1, input_size).contiguous(),
			 bunch_of_grads.slice(0, 0, state_size_3),
			 bunch_of_grads.slice(0, state_size_3, gate_size + state_size_3),
			 bunch_of_grads.slice(0, gate_size + state_size_3, gate_size * 2),
			 bunch_of_grads.slice(0, gate_size * 2, gate_size * 2 + state_size),
			 bunch_of_grads.slice(0, gate_size * 2 + state_size, gate_size * 2 + state_size_2),
			 bunch_of_grads.slice(0, gate_size * 2 + state_size_2, gate_size * 2 + state_size_3),
			 bunch_of_grads.slice(0, gate_size * 2 + state_size_3, gate_size * 3),
			 bunch_of_grads.slice(0, gate_size * 3),
			 grad_hidden,
			 grad_cell };
}
