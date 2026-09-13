"""Local logit-Jacobian diagnostic for method 2a, not an accuracy experiment.

Called by benchmark_tc_checkpointed_training.py --sensitivity-projections N.
Rademacher logit projections estimate full Jacobian Frobenius inner products.
No weights are updated. Numerical step selection and sampled defects are held
fixed in the backward path, as in the current production training method.
"""
import json
from pathlib import Path
import types
import torch


def alignment_metrics(g, p):
    # g: [projections, codes, voltage_grid]; flattened random logit projections.
    vectors=g.permute(1,0,2).flatten(1).double()
    total=vectors.sum(0)
    pred=p.double()[:,None]*total
    energy=vectors.square().sum()
    total_energy=total.square().sum()
    residual=(vectors-pred).square().sum()
    dots=vectors@total
    norms=vectors.norm(dim=1)*total.norm()
    cos=dots/norms.clamp_min(1e-300)
    fit=dots/total_energy.clamp_min(1e-300)
    free_residual=(vectors-fit[:,None]*total).square().sum()
    active=p>0
    return dict(relative_residual=float((residual/energy.clamp_min(1e-300)).sqrt()),
        best_coefficient_residual=float((free_residual/energy.clamp_min(1e-300)).sqrt()),
        jacobian_energy=float(energy/g.shape[0]),total_energy=float(total_energy/g.shape[0]),
        residual_energy=float(residual/g.shape[0]),
        code_cosines=[float(v) if bool(a) and float(n)>0 else None for v,a,n in zip(cos,active,norms)],
        fitted_coefficients=fit.tolist(),histogram=p.tolist(),
        active_code_count=int(active.sum()))


def run_sensitivity(model,x,projections,output,metadata):
    model.eval()
    for param in model.parameters():param.requires_grad_(False)
    package=model.PcConvs[0]._tc_curve_package
    generator=torch.Generator(device=x.device).manual_seed(722)
    # Estimate normalized-distortion means, not R/mean(R). The same prepared
    # physical package is used in all tensors; use MC only once here.
    count=package.means.shape[0]
    codes=torch.arange(1,count+1,device=x.device).repeat(4096,1)
    with torch.no_grad():
        resistances=package.sample(codes,generator=generator)
        distortions=package.programmed_resistances[None,:,None]/resistances
        distortion_means=distortions.mean(0)
        centered=distortions-distortion_means
        distortion_covariances=torch.einsum('slv,slw->lvw',centered,centered)/(distortions.shape[0]-1)
        del distortions,centered
    del resistances,codes
    entries=[]
    for layer,block in enumerate(model.PcConvs):
        curves={}
        for name in ('FFconv','FBconv'):
            c=block._values_to_level_idx(getattr(block,name).weight.detach())
            hist=torch.bincount(c.flatten(),minlength=count+1)[1:].to(x)
            p=hist/hist.sum().clamp_min(1)
            mean=(p[:,None]*distortion_means).sum(0)
            d=mean.repeat(count,1).detach().requires_grad_()
            curves[name]=package.programmed_resistances[:,None]/d
            entries.append(dict(layer=layer,tensor=name,p=p,d=d))
        block._tc_conv_method='grouped'
        # Capture the same differentiable curves throughout search/replay.
        block._tc_curves_for_solve=types.MethodType(lambda self,_curves=curves:_curves,block)
    logits=model(x)
    tensors=[e['d'] for e in entries]
    accumulated=[[] for _ in entries]
    probe_generator=torch.Generator(device=x.device).manual_seed(811)
    for index in range(projections):
        signs=torch.randint(0,2,logits.shape,device=x.device,generator=probe_generator).to(logits)*2-1
        gradients=torch.autograd.grad(logits,tensors,grad_outputs=signs,
            retain_graph=index<projections-1,allow_unused=True)
        for collected,d,gradient in zip(accumulated,tensors,gradients):
            collected.append(torch.zeros_like(d) if gradient is None else gradient.detach())
        print(f'Sensitivity projection {index+1}/{projections}',flush=True)
    results=[]
    for entry,gs in zip(entries,accumulated):
        row=dict(layer=entry['layer'],tensor=entry['tensor'])
        g=torch.stack(gs);p=entry['p']
        row.update(alignment_metrics(g,p))
        # Also test only distortion directions actually excited by the mixture.
        # If K_mix=A A^T, this compares J_l A with p_l (sum J_l) A.
        mean=(p[:,None]*distortion_means).sum(0)
        delta=distortion_means-mean
        covariance=(p[:,None,None]*distortion_covariances).sum(0)+torch.einsum('l,lv,lw->vw',p,delta,delta)
        eigenvalues,eigenvectors=torch.linalg.eigh(covariance.double())
        factor=eigenvectors*eigenvalues.clamp_min(0).sqrt()
        weighted=g.double()@factor
        row['mixture_weighted']=alignment_metrics(weighted,p)
        results.append(row)
    energy=sum(r['jacobian_energy'] for r in results)
    residual=sum(r['residual_energy'] for r in results)
    significant=[r for r in results if r['jacobian_energy']>energy*1e-6]
    result=dict(**metadata,projections=projections,distortion_mean_mc_draws=4096,
        physical_defects='configured defects active; one fixed forward realization (see mode)',
        dropout=False,weight_updates=False,unrolled=False,method='exact grouped dense operator',
        comparison='J_code versus p_code * sum_code J_code; random-logit-projection estimate',
        global_relative_residual=(residual/max(energy,1e-300))**.5,
        mixture_weighted_relative_residual=(sum(r['mixture_weighted']['residual_energy'] for r in results)/
            max(sum(r['mixture_weighted']['jacobian_energy'] for r in results),1e-300))**.5,
        significant_tensor_count=len(significant),finite_logits=bool(logits.isfinite().all()),
        rows=results,peak_allocated=torch.cuda.max_memory_allocated())
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
