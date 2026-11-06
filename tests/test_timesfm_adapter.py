"""
TimesFM adaptoru — hizalama, sizinti, onbellek ve varsayilan-kapali bekcisi

Gercek 330M modeli YUKLEMEZ: sahte bir tahminci baglami kaydeder ve son
degerden turetilmis dilimler dondurur. Boylece su sorular kesin yanitlanir:

  - Satir i'nin baglami X[.., i]'de mi bitiyor (gelecek satir sizmiyor mu)?
  - Gunluk hedef ufkun 2. adimi mi (baglam r_{t-1}'de bitiyor, hedef r_{t+1})?
  - Ek degiskenlerin sekli dogru mu, eksik kolon sessizce mi atlaniyor?
  - Ensemble'da varsayilan KAPALI mi, trainer/HPO varsayilanlarina sizmiyor mu?
  - predict_next'in yeni `_predict_last` yolu mevcut modeller icin ayni mi?

Kullanim:
    python tests/test_timesfm_adapter.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)

import numpy as np

import prediction.models.timesfm_model as tm
from prediction.models.base import BasePredictionModel
from prediction.models.timesfm_model import TimesFMModel

FAILS = []


def check(name, cond, detail=''):
    print(('  [OK]   ' if cond else '  [FAIL] ') + name + (f'  {detail}' if detail else ''))
    if not cond:
        FAILS.append(name)


class _Out:
    def __init__(self, quantiles):
        self.quantiles = quantiles


class FakeForecaster:
    """Adim j medyani = baglamin son degeri + j * 1e-3; bant +-0.01 * |k-4|."""

    def __init__(self):
        self.calls = []

    def predict_batch(self, contexts, horizon, past_only_covariates=None,
                      past_future_covariates=None, return_quantiles=True, **_):
        self.calls.append({'contexts': contexts, 'horizon': horizon,
                           'past': past_only_covariates, 'future': past_future_covariates})
        for ctx in contexts:
            last = float(np.atleast_2d(ctx)[0, -1])
            med = last + np.arange(horizon) * 1e-3
            q = med[:, None] + (np.arange(9) - 4)[None, :] * 0.01
            yield _Out(q[None, :, :])


COLS = ['log_return_1d', 'macro_policy_rate', 'macro_cpi', 'macro_usdtry_return',
        'day_of_week', 'month', 'other']


def make_X(n=200):
    X = np.zeros((n, len(COLS)), dtype=np.float32)
    X[:, 0] = np.arange(n)            # hedef kolonu = satir numarasi (sizinti testi icin)
    X[:, 1] = 40.0
    X[:, 2] = 30.0
    X[:, 3] = 0.001
    X[:, 4] = np.arange(n) % 5        # Pzt..Cum
    X[:, 5] = 6
    return X


def make_model(fake, horizon='daily', **over):
    tm.clear_cache()
    tm._get_forecaster = lambda c, d: fake
    params = {**TimesFMModel().get_default_params(), 'context_len': 50, 'min_context': 10,
              'batch_size': 16, **over}
    m = TimesFMModel(horizon=horizon, params=params)
    m.feature_cols = list(COLS)
    m._build_model(m.params)
    m._fit(None, None, None, None)
    return m


def main():
    print('\n[1] Varsayilan kapali, trainer/HPO varsayilanlarina girmiyor')
    from prediction.models import MODEL_REGISTRY
    from prediction.models.ensemble import BASE_MODEL_CONFIGS, StackingEnsemble
    check('ensemble config: timesfm enabled=False',
          BASE_MODEL_CONFIGS.get('timesfm', {}).get('enabled') is False)
    check('MODEL_REGISTRY timesfm icermiyor (trainer/HPO varsayilani degismez)',
          'timesfm' not in MODEL_REGISTRY)
    check('ensemble timesfm sinifini cozuyor',
          StackingEnsemble()._get_model_class('timesfm') is TimesFMModel)
    enabled = [k for k, v in BASE_MODEL_CONFIGS.items() if v.get('enabled', True)]
    check('etkin modeller degismedi', enabled == ['xgboost', 'lightgbm', 'catboost', 'bilstm', 'tft'],
          str(enabled))

    print('\n[2] Baglam hizalamasi ve sizinti')
    fake = FakeForecaster()
    m = make_model(fake)
    X = make_X(200)
    q = m.predict_quantiles(X)
    check('cikti uzunlugu n - min_context + 1', len(q) == 200 - 10 + 1, str(len(q)))
    ctxs = [c for call in fake.calls for c in call['contexts']]
    ok_end = all(int(c[0, -1]) == 9 + j for j, c in enumerate(ctxs))
    check('satir i baglami X[i]\'de bitiyor', ok_end)
    check('hicbir baglam gelecek satir icermiyor',
          all(c[0].max() == c[0, -1] for c in ctxs))
    check('baglam uzunlugu context_len ile sinirli', max(c.shape[1] for c in ctxs) == 50)
    check('ilk tahmin min_context satirla', ctxs[0].shape[1] == 10)
    check('ufuk = 2 adim (r_t atlanir, r_{t+1} hedef)', fake.calls[0]['horizon'] == 2)
    check('gunluk nokta = 2. adim medyani', np.isclose(q[0, 4], 9 + 1e-3), f'{q[0, 4]}')
    check('dilim siralamasi korunuyor', np.all(np.diff(q[0]) > 0))

    print('\n[3] Ek degiskenler')
    call = fake.calls[0]
    past, fut = call['past'][0], call['future'][0]
    L = ctxs[0].shape[1]
    check('gecmis ek degisken sekli (3, L)', past is not None and past.shape == (3, L), str(past.shape))
    check('takvim sekli (4, L + ufuk)', fut is not None and fut.shape == (4, L + 2), str(fut.shape))
    # Satir 9 -> dow 4 (Cum); gelecek 2 adim Pzt (0), Sal (1)
    dow_future = np.round(np.arctan2(fut[0, -2:], fut[1, -2:]) / (2 * np.pi) * 5) % 5
    check('takvim: Cuma -> Pazartesi, Sali', list(dow_future) == [0.0, 1.0], str(dow_future))

    fake_u = FakeForecaster()
    mu = make_model(fake_u, past_covariates=[], calendar_covariates=False)
    mu.predict_quantiles(X[:30])
    check('tek degiskenli: ek degisken yok',
          all(p is None for p in fake_u.calls[0]['past'])
          and all(f is None for f in fake_u.calls[0]['future']))

    fake_m = FakeForecaster()
    mm = make_model(fake_m, past_covariates=['macro_policy_rate', 'yok_boyle_kolon'])
    mm.predict_quantiles(X[:30])
    check('eksik ek degisken sessizce atlaniyor', fake_m.calls[0]['past'][0].shape[0] == 1)

    bad = TimesFMModel(params={**TimesFMModel().get_default_params(), 'context_len': 50})
    bad.feature_cols = ['other']
    try:
        bad._fit(None, None, None, None)
        check('hedef kolon yoksa hata', False)
    except ValueError:
        check('hedef kolon yoksa hata', True)

    print('\n[4] Onbellek')
    m.predict_quantiles(X)  # [3]'teki make_model onbellegi bosaltti — isit
    n_calls = len(fake.calls)
    q2 = m.predict_quantiles(X)
    check('ayni X ikinci kez: yeni cikarim yok', len(fake.calls) == n_calls)
    check('onbellek ayni sonucu veriyor', np.array_equal(q, q2))
    q_sub = m.predict_quantiles(X[:120])
    check('alt dizi: baglami ayni satirlar onbellekten', len(fake.calls) == n_calls
          and np.array_equal(q_sub[-1], q[110]))

    print('\n[5] predict_next yolu (_predict_last)')
    point, prob = m._predict_last(X)
    check('_predict_last = son satir medyani', np.isclose(point, q[-1, 4]))
    check('_predict_last olasiligi = dilimlerden', np.isclose(
        prob, TimesFMModel.prob_up_from_quantiles(q[-1:])[0]))
    fake_l = FakeForecaster()
    ml = make_model(fake_l)
    ml._predict_last(make_X(500))
    check('_predict_last yalnizca 1 satir hesaplar',
          sum(len(c['contexts']) for c in fake_l.calls) == 1)

    print('\n[6] Yon olasiligi')
    qs = np.array([[-0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04]])
    check('medyan 0 -> P(yukari) = 0.5', np.isclose(TimesFMModel.prob_up_from_quantiles(qs)[0], 0.5))
    check('medyan > 0 -> P(yukari) > 0.5',
          TimesFMModel.prob_up_from_quantiles(qs + 0.01)[0] > 0.5)
    check('olasilik [0.01, 0.99] icinde',
          0.01 <= TimesFMModel.prob_up_from_quantiles(qs + 10)[0] <= 0.99)

    print('\n[7] Haftalik ufuk')
    fake_w = FakeForecaster()
    mw = make_model(fake_w, horizon='weekly')
    qw = mw.predict_quantiles(X[:30])
    check('haftalik ufuk 6 adim', fake_w.calls[0]['horizon'] == 6)
    # adim 1..5 medyanlari toplami: 5 * last + (1+2+3+4+5) * 1e-3
    check('haftalik medyan = adim 2..6 toplami', np.isclose(qw[0, 4], 5 * 9 + 15e-3), f'{qw[0, 4]}')

    print('\n[8] Kaydet / yukle (agirlik yok, yalnizca parametre)')
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'tfm')
        m._save_model(p)
        m2 = TimesFMModel(params={'checkpoint': 'x'})
        m2._load_model(p)
        check('parametreler geri geldi', m2.params == m.params)
        check('yuklemede model tembel (None)', m2.model is None)

    print('\n[9] Temel sinif _predict_last mevcut davranisla ayni')

    class Dummy(BasePredictionModel):
        MODEL_TYPE = 'dummy'
        def get_default_params(self): return {}
        def get_optuna_search_space(self, trial): return {}
        def _build_model(self, params): pass
        def _fit(self, *a): return {}
        def _predict_raw(self, X): return X[:, 0] * 2.0
        def _save_model(self, path): pass
        def _load_model(self, path): pass

    class DummyDir(Dummy):
        def _predict_direction_raw(self, X): return X[:, 0] / 1000.0

    Xd = make_X(20)
    pt, pr = Dummy()._predict_last(Xd)
    check('yon basi yoksa olasilik None', pr is None)
    check('nokta = _predict_raw[-1]', pt == float(Dummy()._predict_raw(Xd)[-1]))
    pt2, pr2 = DummyDir()._predict_last(Xd)
    check('yon basi varsa = _predict_direction_raw[-1]', pr2 == float(Xd[-1, 0] / 1000.0))

    print('\n' + '=' * 60)
    if FAILS:
        print(f'BASARISIZ: {len(FAILS)} kontrol -> {FAILS}')
        return 1
    print('TUM KONTROLLER GECTI — TimesFM adaptoru')
    return 0


if __name__ == '__main__':
    sys.exit(main())
