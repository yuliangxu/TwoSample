# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Selected definitions from torchdyn 1.0.6; see ../SOURCE.json.
# Loaded by AST to avoid unrelated optional package dependencies.

def odeint(f:Callable, x:Tensor, t_span:Union[List, Tensor], solver:Union[str, nn.Module], atol:float=1e-3, rtol:float=1e-3,
		   t_stops:Union[List, Tensor, None]=None, verbose:bool=False, interpolator:Union[str, Callable, None]=None, return_all_eval:bool=False,
		   save_at:Union[Iterable, Tensor]=(), args:Dict={}, seminorm:Tuple[bool, Union[int, None]]=(False, None)) -> Tuple[Tensor, Tensor]:
	"""Solve an initial value problem (IVP) determined by function `f` and initial condition `x`.

	   Functional `odeint` API of the `torchdyn` package.

	Args:
		f (Callable):
		x (Tensor):
		t_span (Union[List, Tensor]):
		solver (Union[str, nn.Module]):
		atol (float, optional): Defaults to 1e-3.
		rtol (float, optional): Defaults to 1e-3.
		t_stops (Union[List, Tensor, None], optional): Defaults to None.
		verbose (bool, optional): Defaults to False.
		interpolator (bool, optional): Defaults to False.
		return_all_eval (bool, optional): Defaults to False.
		save_at (Union[List, Tensor], optional): Defaults to t_span
		args (Dict): Arbitrary parameters used in step
		seminorm (Tuple[bool, Union[int, None]], optional): Whether to use seminorms in local error computation.

	Returns:
		Tuple[Tensor, Tensor]: returns a Tuple (t_eval, solution).
	"""
	if t_span[1] < t_span[0]: # time is reversed
		if verbose: warn("You are integrating on a reversed time domain, adjusting the vector field automatically")
		f_ = lambda t, x: -f(-t, x)
		t_span = -t_span
	else: f_ = f

	if type(t_span) == list: t_span = torch.cat(t_span)
	# instantiate the solver in case the user has specified preference via a `str` and ensure compatibility of device ~ dtype
	if type(solver) == str:
		solver = str_to_solver(solver, x.dtype)
	x, t_span = solver.sync_device_dtype(x, t_span)
	stepping_class = solver.stepping_class

	# instantiate the interpolator similar to the solver steps above
	if isinstance(solver, Tsitouras45):
		if verbose: warn("Running interpolation not yet implemented for `tsit5`")
		interpolator = None

	if type(interpolator) == str:
		interpolator = str_to_interp(interpolator, x.dtype)
		x, t_span = interpolator.sync_device_dtype(x, t_span)

	# access parallel integration routines with different t_spans for each sample in `x`.
	if len(t_span.shape) > 1:
		raise NotImplementedError("Parallel routines not implemented yet, check experimental versions of `torchdyn`")
	# odeint routine with a single t_span for all samples
	elif len(t_span.shape) == 1:
		if stepping_class == 'fixed':
			if atol != odeint.__defaults__[0] or rtol != odeint.__defaults__[1]:
				warn("Setting tolerances has no effect on fixed-step methods")
			# instantiate save_at tensor
			return _fixed_odeint(f_, x, t_span, solver, save_at=save_at, args=args)
		elif stepping_class == 'adaptive':
			t = t_span[0]
			k1 = f_(t, x)
			dt = init_step(f, k1, x, t, solver.order, atol, rtol)
			if len(save_at) > 0: warn("Setting save_at has no effect on adaptive-step methods")
			return _adaptive_odeint(f_, k1, x, dt, t_span, solver, atol, rtol, args, interpolator, return_all_eval, seminorm)


def _adaptive_odeint(f, k1, x, dt, t_span, solver, atol=1e-4, rtol=1e-4, args=None, interpolator=None, return_all_eval=False, seminorm=(False, None)):
	"""Adaptive ODE solve routine, called by `odeint`.

	Args:
		f ([type]):
		k1 ([type]):
		x ([type]):
		dt ([type]):
		t_span ([type]):
		solver ([type]):
		atol ([type], optional): Defaults to 1e-4.
		rtol ([type], optional): Defaults to 1e-4.
		args (Dict):
		use_interp (bool, optional):
		return_all_eval (bool, optional): Defaults to False.


	Notes:
		(1) We check if the user wants all evaluated solution points, not only those
		corresponding to times in `t_span`. This is automatically set to `True` when `odeint`
		is called for interpolated adjoints
	"""
	t_eval, t, T = t_span[1:], t_span[:1], t_span[-1]
	ckpt_counter, ckpt_flag = 0, False
	eval_times, sol = [t], [x]
	while t < T:
		if t + dt > T:
			dt = T - t
		############### checkpointing ###############################
		if t_eval is not None:
			# satisfy checkpointing by using interpolation scheme or resetting `dt`
			if (ckpt_counter < len(t_eval)) and (t + dt > t_eval[ckpt_counter]):
				if interpolator == None:
					# save old dt, raise "checkpoint" flag and repeat step
					dt_old, ckpt_flag = dt, True
					dt = t_eval[ckpt_counter] - t

		f_new, x_new, x_err, stages = solver.step(f, x, t, dt, k1=k1, args=args)
		################# compute error #############################
		if seminorm[0] == True:
			state_dim = seminorm[1]
			error = x_err[:state_dim]
			error_scaled = error / (atol + rtol * torch.max(x[:state_dim].abs(), x_new[:state_dim].abs()))
		else:
			error = x_err
			error_scaled = error / (atol + rtol * torch.max(x.abs(), x_new.abs()))
		error_ratio = hairer_norm(error_scaled)
		accept_step = error_ratio <= 1

		if accept_step:
			############### checkpointing via interpolation ###############################
			if t_eval is not None and interpolator is not None:
				coefs = None
				while (ckpt_counter < len(t_eval)) and (t + dt > t_eval[ckpt_counter]):
					t0, t1 = t, t + dt
					x_mid = x + dt * sum([interpolator.bmid[i] * stages[i] for i in range(len(stages))])
					f0, f1, x0, x1 = k1, f_new, x, x_new
					if coefs == None: coefs = interpolator.fit(dt, f0, f1, x0, x1, x_mid)
					x_in = interpolator.evaluate(coefs, t0, t1, t_eval[ckpt_counter])
					sol.append(x_in)
					eval_times.append(t_eval[ckpt_counter][None])
					ckpt_counter += 1

			if t + dt == t_eval[ckpt_counter] or return_all_eval: # note (1)
				sol.append(x_new)
				eval_times.append(t + dt)
				# we only increment the ckpt counter if the solution points corresponds to a time point in `t_span`
				if t + dt == t_eval[ckpt_counter]: ckpt_counter += 1
			t, x = t + dt, x_new
			k1 = f_new

		################ stepsize control ###########################
		# reset "dt" in case of checkpoint without interp
		if ckpt_flag:
			dt = dt_old - dt
			ckpt_flag = False

		dt = adapt_step(dt, error_ratio,
						solver.safety,
						solver.min_factor,
						solver.max_factor,
						solver.order)
	return torch.cat(eval_times), torch.stack(sol)
