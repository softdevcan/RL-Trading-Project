"""
Faz 6 (3.1) — TFT Gruplanmis VSN Esdegerlik Testi

`_FastVariableSelectionNetwork`, degisken bazli GRN dongusunu tek batched
isleme cevirir. Bu testin isi tek bir soruyu kesin yanitlamak:

    Gruplanmis surum, dongulu surumle AYNI fonksiyonu mu hesapliyor?

Ayni agirliklar yuklenip dropout kapatildiginda (eval) ciktilar birebir
esitse, egitim sonuclarindaki fark matematikten degil rastgelelikten
(dropout maskesi / init RNG tuketimi) geliyordur.

Kullanim:
    python tests/test_tft_fast_vsn.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)

import torch

from prediction.models.tft_model import (
    _FastVariableSelectionNetwork,
    _TFTNet,
    _VariableSelectionNetwork,
)

FAILS = []


def check(name, cond, detail=''):
    print(('  [OK]   ' if cond else '  [FAIL] ') + name + (f'  {detail}' if detail else ''))
    if not cond:
        FAILS.append(name)


def copy_weights(slow: _VariableSelectionNetwork, fast: _FastVariableSelectionNetwork):
    """Dongulu VSN'in agirliklarini gruplanmis surume tasi."""
    with torch.no_grad():
        fast.grn_weights.load_state_dict(slow.grn_weights.state_dict())
        g = fast.grn_vars
        for i, grn in enumerate(slow.grn_var):
            g.w1[i].copy_(grn.fc1.weight)
            g.b1[i].copy_(grn.fc1.bias)
            g.w2[i].copy_(grn.fc2.weight)
            g.b2[i].copy_(grn.fc2.bias)
            g.w_fc[i].copy_(grn.glu.fc.weight)
            g.b_fc[i].copy_(grn.glu.fc.bias)
            g.w_gate[i].copy_(grn.glu.gate.weight)
            g.b_gate[i].copy_(grn.glu.gate.bias)
            g.w_skip[i].copy_(grn.skip.weight)
            g.b_skip[i].copy_(grn.skip.bias)
            g.ln_weight[i].copy_(grn.layer_norm.weight)
            g.ln_bias[i].copy_(grn.layer_norm.bias)


def main():
    torch.manual_seed(1234)
    B, T, V, H = 8, 30, 17, 16

    print("1) VSN esdegerligi (ayni agirlik, eval modu)")
    slow = _VariableSelectionNetwork(input_size=V, n_vars=V, hidden_size=H, dropout=0.1).eval()
    fast = _FastVariableSelectionNetwork(input_size=V, n_vars=V, hidden_size=H, dropout=0.1).eval()
    copy_weights(slow, fast)

    x = torch.randn(B, T, V)
    with torch.no_grad():
        out_slow = slow(x)
        out_fast = fast(x)

    check('cikti sekli ayni', out_slow.shape == out_fast.shape,
          f'{tuple(out_slow.shape)} vs {tuple(out_fast.shape)}')
    max_diff = (out_slow - out_fast).abs().max().item()
    check('degerler esit (atol=1e-5)', torch.allclose(out_slow, out_fast, atol=1e-5),
          f'max fark={max_diff:.3e}')

    print("\n2) Gradyan akisi (backward calisiyor)")
    x2 = torch.randn(B, T, V, requires_grad=True)
    fast.train()
    loss = fast(x2).sum()
    loss.backward()
    check('girdi gradyani uretildi', x2.grad is not None and torch.isfinite(x2.grad).all())
    check('parametre gradyanlari sonlu',
          all(p.grad is None or torch.isfinite(p.grad).all() for p in fast.parameters()))

    print("\n3) Kismi degisken kapagi (V < n_vars)")
    fast_cap = _FastVariableSelectionNetwork(
        input_size=V, n_vars=V, hidden_size=H, dropout=0.0).eval()
    x3 = torch.randn(B, T, V - 5)
    try:
        with torch.no_grad():
            out_cap = fast_cap.grn_vars(x3)
        check('dilimlenmis girdi calisiyor', out_cap.shape == (B, T, V - 5, H),
              f'{tuple(out_cap.shape)}')
    except Exception as exc:
        check('dilimlenmis girdi calisiyor', False, f'{type(exc).__name__}: {exc}')

    print("\n4) _TFTNet varyant secimi ve parametre sayisi")
    net_slow = _TFTNet(input_size=V, hidden_size=H, num_heads=2, fast_vsn=False)
    net_fast = _TFTNet(input_size=V, hidden_size=H, num_heads=2, fast_vsn=True)
    n_slow = sum(p.numel() for p in net_slow.parameters())
    n_fast = sum(p.numel() for p in net_fast.parameters())
    check('varsayilan dongulu VSN', isinstance(net_slow.vsn, _VariableSelectionNetwork))
    check('fast_vsn=True gruplanmis VSN', isinstance(net_fast.vsn, _FastVariableSelectionNetwork))
    check('parametre sayisi ayni', n_slow == n_fast, f'{n_slow} vs {n_fast}')

    print("\n5) Ag ciktisi esdegerligi (agirliklar kopyalanmis, eval)")
    net_slow.eval(); net_fast.eval()
    with torch.no_grad():
        # VSN disindaki tum katmanlar ayni isimli -> dogrudan kopyalanir.
        shared = {k: v for k, v in net_slow.state_dict().items() if not k.startswith('vsn.')}
        missing = net_fast.load_state_dict(shared, strict=False)
        copy_weights(net_slow.vsn, net_fast.vsn)
        xn = torch.randn(4, T, V)
        p_slow, d_slow = net_slow(xn)
        p_fast, d_fast = net_fast(xn)
    check('fiyat ciktisi esit', torch.allclose(p_slow, p_fast, atol=1e-5),
          f'max fark={(p_slow - p_fast).abs().max().item():.3e}')
    check('yon ciktisi esit', torch.allclose(d_slow, d_fast, atol=1e-5),
          f'max fark={(d_slow - d_fast).abs().max().item():.3e}')
    check('eksik anahtar yok (vsn disi)', len(missing.unexpected_keys) == 0,
          f'{missing.unexpected_keys[:3]}')

    print('\n' + '=' * 60)
    if FAILS:
        print(f'BASARISIZ: {len(FAILS)} kontrol -> {FAILS}')
        return 1
    print('TUM KONTROLLER GECTI — gruplanmis VSN ayni fonksiyonu hesapliyor')
    return 0


if __name__ == '__main__':
    sys.exit(main())
