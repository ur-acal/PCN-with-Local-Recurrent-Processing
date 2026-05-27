import torch
import torch.nn.functional as F


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
        sanity_check_solver=False,
        sanity_tol=1e-4,
        sanity_check_layerwise=True,
        sanity_check_layerwise_batches=5,
    ):
        self.model = model.to(device).eval()
        self.test_loader = test_loader
        self.device = device

        self.sigma = sigma
        self.n_steps = int(n_steps)
        self.n_seeds = n_seeds
        self.mismatch_type = mismatch_type
        self.add_scale = add_scale
        self.include_bypass_mismatch = include_bypass_mismatch
        self.clamp = clamp
        self.eps = eps

        self.sanity_check_solver = sanity_check_solver
        self.sanity_tol = sanity_tol
        self.sanity_check_layerwise = sanity_check_layerwise
        self.sanity_check_layerwise_batches = sanity_check_layerwise_batches

        if self.sanity_check_layerwise:
            self.sanity_check_layerwise_accuracy(
                max_batches=self.sanity_check_layerwise_batches
            )

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

        x = self._apply_stem_if_needed(x)

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
    # Sanity check: normal forward vs explicit layerwise forward
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _apply_stem_if_needed(self, x):
        """
        Apply optional first plain conv used by PCNetSeparable / PCNetWith1stConv.
        """
        if hasattr(self.model, "first_conv"):
            x = F.relu(self.model.first_conv(x))
        return x

    @torch.no_grad()
    def _manual_block_forward(self, block, x):
        ys, _, _ = self._manual_euler_trajectory(block, x)
        return self._output_transform(block, ys[-1])

    @torch.no_grad()
    def layerwise_forward(self, x):
        """
        Reproduce model.forward, but each ODE block is evaluated by the analyzer's
        manual Euler rollout instead of block.forward().
        """
        x = x.to(self.device)
        self.model.eval()

        x = self._apply_stem_if_needed(x)

        for i in range(self.model.num_layers):
            if getattr(self.model, "BNs", None) is not None:
                x = self.model.BNs[i](x)

            x = self._manual_block_forward(self.model.PcConvs[i], x)

            if self.model.max_pool[i]:
                x = self.model.max_pool2d(x)

            if self.clamp:
                x = torch.clamp(x, -1, 1)

        if self.model.dropout > 0.0:
            x = F.dropout(input=x, p=self.model.dropout, training=self.model.training)

        if getattr(self.model, "BNend", None) is not None:
            feat = self.model.relu(self.model.BNend(x))
        else:
            feat = F.relu(x)

        out = F.avg_pool2d(feat, feat.size(-1))
        out = out.view(out.size(0), -1)
        out = self.model.linear(out)

        return out

    @torch.no_grad()
    def sanity_check_layerwise_accuracy(self, max_batches=5):
        self.model.eval()

        total = 0
        correct_model = 0
        correct_layerwise = 0

        max_abs_logit_diff = 0.0
        mean_abs_logit_diff_sum = 0.0
        n_batches_used = 0

        for batch_idx, batch in enumerate(self.test_loader):
            if batch_idx >= max_batches:
                break

            images, labels = batch[:2]
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            logits_model = self.model(images)
            logits_layerwise = self.layerwise_forward(images)

            pred_model = logits_model.argmax(dim=1)
            pred_layerwise = logits_layerwise.argmax(dim=1)

            total += labels.numel()
            correct_model += pred_model.eq(labels).sum().item()
            correct_layerwise += pred_layerwise.eq(labels).sum().item()

            diff = (logits_model - logits_layerwise).abs()
            max_abs_logit_diff = max(max_abs_logit_diff, diff.max().item())
            mean_abs_logit_diff_sum += diff.mean().item()
            n_batches_used += 1

        acc_model = correct_model / max(total, 1)
        acc_layerwise = correct_layerwise / max(total, 1)
        mean_abs_logit_diff = mean_abs_logit_diff_sum / max(n_batches_used, 1)

        out = {
            "acc_model": acc_model,
            "acc_layerwise": acc_layerwise,
            "acc_diff": acc_layerwise - acc_model,
            "max_abs_logit_diff": max_abs_logit_diff,
            "mean_abs_logit_diff": mean_abs_logit_diff,
            "n_batches": n_batches_used,
            "n_samples": total,
        }

        print("Layerwise accuracy sanity check:")
        for key, value in out.items():
            print(f"{key}: {value}")

        return out

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _analyze_ode_block(self, block, x):
        block.eval()

        params = self._unique_weight_params(block)

        # Clean trajectory from block's own solver.
        ys_clean, h, T = self._solver_trajectory(block, x)

        # Sanity check: block.forward_full_steps vs manual Euler.
        solver_manual_rel_err = None
        solver_manual_abs_err = None

        if self.sanity_check_solver:
            ys_manual, _, _ = self._manual_euler_trajectory(block, x)

            solver_manual_abs_err_tensor = (
                ys_clean[-1] - ys_manual[-1]
            ).flatten(1).norm(dim=1).mean()

            manual_ref = ys_manual[-1].flatten(1).norm(dim=1).mean().clamp_min(self.eps)
            solver_manual_rel_err = (solver_manual_abs_err_tensor / manual_ref).item()
            solver_manual_abs_err = solver_manual_abs_err_tensor.item()

            if solver_manual_rel_err > self.sanity_tol:
                print(
                    "WARNING: block.forward_full_steps and manual Euler differ. "
                    f"rel_err={solver_manual_rel_err:.3e}, "
                    f"abs_err={solver_manual_abs_err:.3e}"
                )

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
            # 1. Actual full fixed-mismatch ODE sensitivity.
            # Same frozen mismatch used for whole ODE solve.
            # ----------------------------------------------------------
            self._add_param_perturb(params, deltas)

            ys_pert, _, _ = self._solver_trajectory(block, x)
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
            # 3. Stepwise contributions v_k.
            # Mismatch only at step k, then clean propagation.
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

        C_from_H_summary = self._summarize_C(C_from_H, "C_from_H")
        C_avg_cos_summary = self._summarize_C(C_avg_cos, "C_avg_cos")

        D_ode_mean = torch.cat(D_ode_vals).mean()
        D_one_step_mean = torch.cat(D_one_step_vals).mean()

        return {
            "D_ode": D_ode_mean.item(),
            "D_one_step": D_one_step_mean.item(),
            "D_ratio_ode_over_one_step": (
                D_ode_mean / D_one_step_mean.clamp_min(self.eps)
            ).item(),
            "rho": torch.cat(rho_vals).mean().item(),
            "eta": torch.cat(eta_vals).mean().item(),
            "H": H.detach().cpu(),
            "C_from_H": C_from_H.detach().cpu(),
            "C_avg_cos": C_avg_cos.detach().cpu(),
            # scalar summaries
            **C_from_H_summary,
            **C_avg_cos_summary,

            "n_steps": self.n_steps,
            "sigma": self.sigma,
            "n_seeds": self.n_seeds,
            "mismatch_type": self.mismatch_type,
            "solver_manual_rel_err": solver_manual_rel_err,
            "solver_manual_abs_err": solver_manual_abs_err,
        }

    # ------------------------------------------------------------------
    # Fixed-step Euler utilities
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _solver_trajectory(self, block, x):
        """
        Get trajectory using block.forward_full_steps(x).

        Assumption:
            block.forward_full_steps(x) returns raw ODE states with shape
            [n_steps + 1, B, C, H, W].

        Note:
            Your current forward_full_steps applies bypass if bypass is not None.
            That is not a raw ODE trajectory, so this analyzer rejects bypass blocks.
        """
        if getattr(block, "bypass", None) is not None:
            raise ValueError(
                "block.forward_full_steps() applies bypass when bypass is not None. "
                "For this analyzer, use blocks without bypass or add a raw trajectory method."
            )

        block.integration_time = block.integration_time.type_as(x)

        out = block.forward_full_steps(x)[0][0]

        if not torch.is_tensor(out):
            out = torch.stack(list(out), dim=0)

        t0 = block.integration_time[0].to(device=x.device, dtype=x.dtype)
        t1 = block.integration_time[-1].to(device=x.device, dtype=x.dtype)

        T = t1 - t0
        h = T / self.n_steps

        y0 = block.init_y(x)
        out0 = out[0]

        def rel_err(a, b):
            return (a - b).flatten(1).norm(dim=1).mean() / b.flatten(1).norm(dim=1).mean().clamp_min(self.eps)

        first_vs_y0 = None
        first_vs_x = None

        if out0.shape == y0.shape:
            first_vs_y0 = rel_err(out0, y0).item()

        if out0.shape == x.shape:
            first_vs_x = rel_err(out0, x).item()

        # If solver returns n_steps states, we expect [y1, ..., yN], not y0.
        if out.shape[0] == self.n_steps:
            if first_vs_y0 is not None and first_vs_y0 < 1e-7:
                raise ValueError(
                    "block.forward_full_steps returned n_steps states, but out[0] appears to be y0. "
                    f"first_vs_y0={first_vs_y0:.3e}. "
                    "This means the solver may already include the initial state; do not prepend y0 blindly."
                )

            if first_vs_x is not None and first_vs_x < 1e-7:
                raise ValueError(
                    "block.forward_full_steps returned n_steps states, but out[0] appears to be x. "
                    f"first_vs_x={first_vs_x:.3e}. "
                    "This is unexpected for an ODE state trajectory."
                )

            # Solver returned [y1, ..., yN], so prepend y0.
            ys = [y0] + [out[i] for i in range(out.shape[0])]

        # If solver returns n_steps+1 states, we expect [y0, y1, ..., yN].
        elif out.shape[0] == self.n_steps + 1:
            if first_vs_y0 is not None and first_vs_y0 > 1e-5:
                raise ValueError(
                    "block.forward_full_steps returned n_steps+1 states, but out[0] does not look like y0. "
                    f"first_vs_y0={first_vs_y0:.3e}."
                )

            ys = [out[i] for i in range(out.shape[0])]

        else:
            raise ValueError(
                f"block.forward_full_steps returned {out.shape[0]} states, "
                f"but analyzer n_steps={self.n_steps}. "
                f"Expected either {self.n_steps} states [y1...yN] "
                f"or {self.n_steps + 1} states [y0...yN]."
            )

        return ys, h, T

    @torch.no_grad()
    def _manual_euler_trajectory(self, block, x):
        """
        Manual Euler rollout used only for sanity check.
        """
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
        """
        Continue clean Euler dynamics from arbitrary intermediate state y_start.

        This remains manual because block.forward_full_steps starts from block.init_y(x),
        not from arbitrary y_start.
        """
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

    @staticmethod
    def _summarize_C(C, prefix):
        """
        Summarize a K x K temporal alignment matrix.

        C[k, l] measures alignment between step-k and step-l mismatch errors.
        """
        C = C.detach()
        K = C.shape[0]
        device = C.device

        offdiag_mask = ~torch.eye(K, dtype=torch.bool, device=device)
        offdiag = C[offdiag_mask]

        if K > 1:
            adjacent = torch.diag(C, diagonal=1)
            first_last = C[0, -1]
        else:
            adjacent = torch.tensor([float("nan")], device=device, dtype=C.dtype)
            first_last = torch.tensor(float("nan"), device=device, dtype=C.dtype)

        return {
            f"{prefix}_offdiag_mean": offdiag.mean().item(),
            f"{prefix}_offdiag_abs_mean": offdiag.abs().mean().item(),
            f"{prefix}_offdiag_min": offdiag.min().item(),
            f"{prefix}_offdiag_max": offdiag.max().item(),
            f"{prefix}_adjacent_mean": adjacent.mean().item(),
            f"{prefix}_first_last": first_last.item(),
            f"{prefix}_negative_frac": (offdiag < 0).float().mean().item(),
        }