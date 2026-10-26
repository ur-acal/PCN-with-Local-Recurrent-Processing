"""Public launcher/default and recovery compatibility for top-n TC curves."""
import pytest

from test_tc_cli import resolved
from test_training_recovery import make_trainer
from training_recovery import HISTORY, latest_path, save_latest, restore_latest


def test_local_pcn_launcher_defaults_and_forwarding():
    assert resolved('ft').tc_curve_top_n == 0
    for n in (1,2,3):
        assert resolved('ft',TC_CURVE_TOP_N=str(n)).tc_curve_top_n == n
        # The same environment is harmless to expanded per-coupler eval.
        assert resolved('eval',TC_CURVE_TOP_N=str(n)).tc_curve_top_n == n


@pytest.mark.parametrize('saved,current,accepted',[(None,0,True),(0,0,True),(2,2,True),(None,1,False),(2,0,False),(1,3,False)])
def test_exact_recovery_top_n(tmp_path,saved,current,accepted):
    trainer=make_trainer(tmp_path)
    config=dict(tc_nonidealities=True)
    if saved is not None:config['tc_curve_top_n']=saved
    trainer.recovery_config=config
    save_latest(trainer,1,dict.fromkeys(HISTORY,None))
    trainer.recovery_checkpoint=latest_path(trainer)
    trainer.recovery_config=dict(config,tc_curve_top_n=current)
    if accepted:restore_latest(trainer)
    else:
        with pytest.raises(ValueError,match='tc_curve_top_n'):restore_latest(trainer)


def test_legacy_empirical_shared_recovery_preserves_full_loop():
    from argparse import Namespace
    from tc_empirical import sampling_mode
    args=Namespace(tc_nonidealities=True,nonlinear_R_curve_sampling='empirical_with_replacement',
        _exact_training_recovery=True,_tc_distribution_version=1,
        tc_conv_method='shared',tc_empirical_curve_bank='/bank')
    sampling_mode(args)
    assert args.tc_conv_method=='loop'
    args.tc_curve_top_n=0
    args.tc_conv_method='shared'
    sampling_mode(args)
    assert args.tc_conv_method=='shared'  # New explicit N=0 recipes remain shared.


def test_top_n_diagnostic_refuses_existing_output_without_launching(tmp_path,monkeypatch):
    import sys
    from unittest.mock import Mock
    from diagnostic_scripts import benchmark_tc_top_n as diagnostic
    launch=Mock(side_effect=AssertionError('Must not launch or reuse stale reports'))
    monkeypatch.setattr(diagnostic.subprocess,'check_output',launch)
    monkeypatch.setattr(sys,'argv',['benchmark_tc_top_n.py','--output',str(tmp_path),
        '--modes','empirical_with_replacement'])
    with pytest.raises(FileExistsError):diagnostic.main()
    launch.assert_not_called()


@pytest.mark.parametrize('flag,value',[('--states','3'),('--ns','-1'),('--modes','unknown')])
def test_top_n_diagnostic_rejects_invalid_selections(flag,value,monkeypatch):
    import sys
    from diagnostic_scripts import benchmark_tc_top_n as diagnostic
    monkeypatch.setattr(sys,'argv',['benchmark_tc_top_n.py',flag,value])
    with pytest.raises(SystemExit) as failure:diagnostic.main()
    assert failure.value.code==2


def test_toggle_operator_diagnostic_uses_current_pinned_pretrain_root():
    from diagnostic_scripts.check_toggle_input_correction import pinned_pretrain_root
    assert pinned_pretrain_root().name=='C100_C96_454_16L_allReLU_digital_biasfalse_iq12_toggle_odexinit'
