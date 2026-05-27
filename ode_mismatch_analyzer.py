import torch


class ODEMismatchAnalyzer:
    def __init__(
        self,
        model,
        test_loader,
        device="cuda",
        sigma=0.3,
        n_steps=50,
        n_seeds=16,
        mismatch_type="mul",   # "mul" or "add"
        add_scale="max_abs",   # "max_abs" or a fixed float
        include_bypass_mismatch=False,
        clamp=False,
        eps=1e-12,
    ):
        self.model = model.to(device).eval()
        self.test_loader = test_loader
        self.device = device

        self.sigma = sigma
        self.n_steps = n_steps
        self.n_seeds = n_seeds
        self.mismatch_type = mismatch_type
        self.add_scale = add_scale
        self.include_bypass_mismatch = include_bypass_mismatch
        self.clamp = clamp
        self.eps = eps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def analyze_block(self, layer_idx, images=None):
        """
        Analyze one ODE block/layer.

        Returns dict with:
            D_ode       : actual full fixed-mismatch ODE layer sensitivity
            D_one_step  : one-step same-vector-field sensitivity
            rho         : accumulation metric, mixes magnitude + direction
            eta         : directional coherence metric
            H           : average raw dot-product matrix
            C_from_H    : theory-style normalized C_kl
            C_avg_cos   : average per-sample/per-seed cosine C_kl
        """
        if images is None:
            images, _ = self._get_one_batch()

        images = images.to(self.device)
        block = self.model.PcConvs[layer_idx]
        x_layer = self.get_input_to_layer(images, layer_idx)

        return self._analyze_ode_block(block, x_layer)

    @torch.no_grad()
    def analyze_all_blocks(self, images=None):
        """
        Analyze all ODE blocks in model.PcConvs.
        """
        if images is None:
            images, _ = self._get_one_batch()

        results = {}
        for layer_idx in range(self.model.num_layers):
            results[layer_idx] = self.analyze_block(layer_idx, images=images)
        return results

    @torch.no_grad()
    def get_input_to_layer(self, x, layer_idx):
        """
        Reproduce model.forward up to the input of PcConvs[layer_idx].
        """
        x = x.to(self.device)
        self.model.eval()

        for i in range(layer_idx):
            if getattr(self.model, "BNs", None) is not None:
                x = self.model.BNs[i](x)

            x = self.model.PcConvs[i](x, i)

            if self.model.max_pool[i]:
                x = self.model.max_pool2d(x)

            if self.clamp:
                x = torch.clamp(x, -1, 1)

        if getattr(self.model, "BNs", None) is not None:
            x = self.model.BNs[layer_idx](x)

        return x

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _analyze_ode_block(self, block, x):
        block.eval()

        params = self._unique_weight_params(block)

        ys_clean, h, T = self._fixed_euler_trajectory(block, x)
        yN_clean_raw = ys_clean[-1]
        yN_clean = self._output_transform(block, yN_clean_raw)
        yN_clean_flat = yN_clean.flatten(1)
        yN_clean_norm = yN_clean_flat.norm(dim=1).clamp_min(self.eps)

        # Clean one-step FF-like baseline: y0 + T F(y0)
        f_clean = block._make_ode_fn(x)
        y0 = ys_clean[0]
        t0 = block.integration_time[0].to(device=x.device, dtype=x.dtype)

        y1_clean_raw = y0 + T * f_clean(t0, y0)
        y1_clean = self._output_transform(block, y1_clean_raw)
        y1_clean_flat = y1_clean.flatten(1)
        y1_clean_norm = y1_clean_flat.norm(dim=1).clamp_min(self.eps)

        D_ode_vals = []
        D_one_step_vals = []
        rho_vals = []
        eta_vals = []

        H_accum = torch.zeros(self.n_steps, self.n_steps, device=x.device, dtype=x.dtype)
        C_avg_cos_accum = torch.zeros_like(H_accum)

        for _ in range(self.n_seeds):
            deltas = self._sample_deltas(params)

            # ----------------------------------------------------------
            # 1. Actual full fixed-mismatch ODE sensitivity
            # ----------------------------------------------------------
            self._add_param_perturb(params, deltas)

            ys_pert, _, _ = self._fixed_euler_trajectory(block, x)
            yN_pert = self._output_transform(block, ys_pert[-1])

            self._remove_param_perturb(params, deltas)

            D_ode_b = (
                (yN_pert.flatten(1) - yN_clean_flat).norm(dim=1)
                / (self.sigma * yN_clean_norm)
            )
            D_ode_vals.append(D_ode_b)

            # ----------------------------------------------------------
            # 2. One-step same-vector-field sensitivity
            # ----------------------------------------------------------
            self._add_param_perturb(params, deltas)

            f_pert = block._make_ode_fn(x)
            y1_pert_raw = y0 + T * f_pert(t0, y0)
            y1_pert = self._output_transform(block, y1_pert_raw)

            self._remove_param_perturb(params, deltas)

            D_one_b = (
                (y1_pert.flatten(1) - y1_clean_flat).norm(dim=1)
                / (self.sigma * y1_clean_norm)
            )
            D_one_step_vals.append(D_one_b)

            # ----------------------------------------------------------
            # 3. Stepwise contributions v_k
            # ----------------------------------------------------------
            v_list = []

            for k in range(self.n_steps):
                yk_clean = ys_clean[k]
                tk = t0 + k * h

                # Apply mismatch only during RHS evaluation at step k.
                self._add_param_perturb(params, deltas)

                f_pert = block._make_ode_fn(x)
                ykp1_pert = yk_clean + h * f_pert(tk, yk_clean)

                self._remove_param_perturb(params, deltas)

                # Then propagate this perturbation to final time cleanly.
                yN_from_k_raw = self._continue_clean_from(
                    block=block,
                    x=x,
                    y_start=ykp1_pert,
                    start_step=k + 1,
                    h=h,
                )
                yN_from_k = self._output_transform(block, yN_from_k_raw)

                # v_k ~= A_k xi
                vk = (yN_from_k.flatten(1) - yN_clean_flat) / self.sigma
                v_list.append(vk)

            # V: [K, B, D]
            V = torch.stack(v_list, dim=0)

            V_sum = V.sum(dim=0)                         # [B, D]
            V_sum_sq = V_sum.pow(2).sum(dim=1)            # [B]

            V_step_sq = V.pow(2).sum(dim=2)               # [K, B]
            V_step_norm = V_step_sq.clamp_min(self.eps).sqrt()

            # rho: mixes directional coherence and magnitude imbalance.
            rho_b = V_sum_sq / (
                self.n_steps * V_step_sq.sum(dim=0).clamp_min(self.eps)
            )

            # eta: directional coherence, equals 1 for any positive aligned v_k.
            eta_b = V_sum_sq / (
                V_step_norm.sum(dim=0).pow(2).clamp_min(self.eps)
            )

            rho_vals.append(rho_b)
            eta_vals.append(eta_b)

            # H_kl = E[(v_k)^T v_l], averaged over batch and seeds.
            H_seed = torch.einsum("kbd,lbd->kl", V, V) / V.shape[1]
            H_accum += H_seed

            # Average cosine version of C_kl.
            Vn = V / V_step_norm.unsqueeze(2).clamp_min(self.eps)
            C_seed_b = torch.einsum("kbd,lbd->bkl", Vn, Vn)
            C_avg_cos_accum += C_seed_b.mean(dim=0)

        H = H_accum / self.n_seeds

        diag = torch.diag(H).clamp_min(self.eps).sqrt()
        C_from_H = H / (diag[:, None] * diag[None, :]).clamp_min(self.eps)

        C_avg_cos = C_avg_cos_accum / self.n_seeds

        return {
            "D_ode": torch.cat(D_ode_vals).mean().item(),
            "D_one_step": torch.cat(D_one_step_vals).mean().item(),
            "D_ratio_ode_over_one_step": (
                torch.cat(D_ode_vals).mean() / torch.cat(D_one_step_vals).mean().clamp_min(self.eps)
            ).item(),
            "rho": torch.cat(rho_vals).mean().item(),
            "eta": torch.cat(eta_vals).mean().item(),
            "H": H.detach().cpu(),
            "C_from_H": C_from_H.detach().cpu(),
            "C_avg_cos": C_avg_cos.detach().cpu(),
            "n_steps": self.n_steps,
            "sigma": self.sigma,
            "n_seeds": self.n_seeds,
            "mismatch_type": self.mismatch_type,
        }

    # ------------------------------------------------------------------
    # Fixed-step Euler utilities
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _fixed_euler_trajectory(self, block, x):
        t0 = block.integration_time[0].to(device=x.device, dtype=x.dtype)
        t1 = block.integration_time[-1].to(device=x.device, dtype=x.dtype)

        T = t1 - t0
        h = T / self.n_steps

        f = block._make_ode_fn(x)
        y = block.init_y(x)

        ys = [y]
        t = t0

        for _ in range(self.n_steps):
            y = y + h * f(t, y)
            ys.append(y)
            t = t + h

        return ys, h, T

    @torch.no_grad()
    def _continue_clean_from(self, block, x, y_start, start_step, h):
        f = block._make_ode_fn(x)

        t0 = block.integration_time[0].to(device=x.device, dtype=x.dtype)
        t = t0 + start_step * h

        y = y_start
        for _ in range(start_step, self.n_steps):
            y = y + h * f(t, y)
            t = t + h

        return y

    # ------------------------------------------------------------------
    # Mismatch utilities
    # ------------------------------------------------------------------

    def _unique_weight_params(self, block):
        params = []

        def add_param(p):
            if p is None:
                return
            if not any(p is q for q in params):
                params.append(p)

        add_param(block.FFconv.weight)

        if hasattr(block, "FBconv") and block.FBconv is not None:
            add_param(block.FBconv.weight)

        if (
            self.include_bypass_mismatch
            and getattr(block, "bypass", None) is not None
        ):
            add_param(block.bypass.weight)

        return params

    def _sample_deltas(self, params):
        deltas = []

        for p in params:
            z = torch.randn_like(p)

            if self.mismatch_type == "mul":
                delta = self.sigma * p * z

            elif self.mismatch_type == "add":
                if self.add_scale == "max_abs":
                    scale = p.detach().abs().max()
                elif isinstance(self.add_scale, (float, int)):
                    scale = float(self.add_scale)
                else:
                    raise ValueError(f"Unknown add_scale={self.add_scale}")

                delta = self.sigma * scale * z

            else:
                raise ValueError(f"Unknown mismatch_type={self.mismatch_type}")

            deltas.append(delta)

        return deltas

    @torch.no_grad()
    def _add_param_perturb(self, params, deltas):
        for p, d in zip(params, deltas):
            p.add_(d)

    @torch.no_grad()
    def _remove_param_perturb(self, params, deltas):
        for p, d in zip(params, deltas):
            p.sub_(d)

    @staticmethod
    def _output_transform(block, y):
        if getattr(block, "bypass", None) is not None:
            return block.bypass(y) + y
        return y

    def _get_one_batch(self):
        return next(iter(self.test_loader))