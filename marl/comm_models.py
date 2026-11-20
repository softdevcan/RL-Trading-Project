"""Faz 10 M4 + M5: iletisim kollu politika modelleri (BenchMARL 1.4 `Model`). YALNIZ `venv-marl/`.

Kollar (plan §3.2.5; M4 tasarimi `docs/development/phase-10-m4-design.md` §1):
  none : BenchMARL MLP (iletisimsiz; ajan basina paylasimli MLP)  -> bu modul KULLANILMAZ
  attn : paylasimli kodlayici -> ajanlar arasi cok basli oz-dikkat (tam bagli; TarMAC tipi, Das vd. 2019)
         -> paylasimli cikti (spike `scripts/spike_m1/benchmarl_glue.py::AttnComm` ile ayni yapi)
  gat  : attn ile AYNI katmanlar ve AYNI parametre sayisi; dikkat yalniz graf komsulari + kendisi ile sinirli
         (sabit ikili maske; graf egitim penceresinden, `scripts/marl/graph_m4.py`). Iki kol arasindaki tek fark
         graf onculudur (plan §2 Y-tablosu #3: "fark graf onculunden gelir").
  self : attn/gat ile AYNI katmanlar ve AYNI parametre; dikkat yalniz kendine (maske = birim) -> ajanlar arasi bilgi
         akisi YOK. M4b mimari-es iletisimsiz kontrol (M4 tasarimi §6; hakem-rl M4a #4): gat - self farki yalniz iletisim.

M5 S-msg (plan §3.1 sinyal yollari; Faz 9 M5; M5 tasarimi §3 TASLAK): guvenle agirlikli dikkat
  puan_ij = q_i k_j / sqrt(d) + beta * b_j,  b_j gondericinin guven terimi (alicidan bagimsiz)
  conf_mode 'prob'     : c_j = clip(0,5 + |p_j - 0,5|, c_min, c_max), b_j = log c_j   (plan; beta = 1 DebUnc)
  conf_mode 'zlogodds' : b_j = |logit p_j| / z_scale  (hakem-2 #10: olcekten bagimsiz; z_scale egitim
                         satirlarindaki |logit p| std'si, disaridan verilir)
  Tahmini olmayan gonderici (pred_ok = 0): b_j = en dusuk deger (log c_min | 0); 0,5'e sessizce dusulmez.
  beta_mode 'fixed' (beta_init sabit; 0 = guvensiz dikkat, 1 = DebUnc) | 'learned' (skaler, beta_init'ten).
  gate: alici oz-guven gecidi g_i = sigmoid(a * d_i + b); d_i alicinin kendi guvensizligi (prob: 1 - c_i, plandaki
        bicim; zlogodds: -|logit p_i| / z_scale). Mesaj g_i ile olceklenir: h <- LN(h + g_i * m_i). Baslangic a = 0,
        b = gate_b_init (4,6 -> g ~ 0,99: gecitli kol gecitsiz kola yakin baslar; tasarim varsayimi).
  shuffle_conf: guven terimi sabit bir ajan permutasyonuyla karistirilir (b_j <- b_pi(j); karistirilmis-guven
        kolu, hakem-2 #10): marjinal dagilim ayni, gonderici-guven baglantisi yok.
  conf_temp: gonderici basina sicaklik T_j (H3 bozulmus-kalibrasyon kolu GLM; M5 tasarimi §3.3, §4): yalniz
        GUVEN TERIMINDE p_j <- sigmoid(T_j * logit p_j); gozlem ve mesaj icerigi DEGISMEZ (kullanici karari
        2026-11-19: fark yalniz mekanizmadan gelsin). Sembol ici siralama ayni, semboller arasi guven sirasi bozulur.
        z_scale bozulmus terimden olculur (`conf_scale(..., temp)`): kuresel olcek GL ile ayni kalir.
Guven gozlemden okunur (`conf_index` = p_up_cal - 0,5 sutunu, `conf_ok_index` = pred_ok); `obs_spec` DEGISMEZ (K2).
conf_mode = 'none' iken model M4'teki CommPolicy ile bit-estir (test_marl_m5_msg).

Yalniz ajan eksenli girdi + tam parametre paylasimi (K24) + merkezi olmayan politika desteklenir; elestirmen
BenchMARL MLP'dir (kollar arasinda ayni).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch
from tensordict import TensorDictBase
from torch import nn

from benchmarl.models.common import Model, ModelConfig

KINDS = ('attn', 'gat', 'self')
CONF_MODES = ('none', 'prob', 'zlogodds')
BETA_MODES = ('fixed', 'learned')


def graph_mask(n_agents: int, edges: Sequence[Sequence[int]]) -> torch.Tensor:
    """Yonsuz kenar listesi -> MultiheadAttention maskesi [N, N], True = ENGELLI. Kendine dikkat her zaman acik
    (her satirda en az bir izinli giris; softmax NaN vermez)."""
    allowed = torch.eye(n_agents, dtype=torch.bool)
    for e in edges:
        i, j = int(e[0]), int(e[1])
        if not (0 <= i < n_agents and 0 <= j < n_agents) or i == j:
            raise ValueError(f"gecersiz kenar {e} (N = {n_agents})")
        allowed[i, j] = allowed[j, i] = True
    return ~allowed


def derangement(n: int, seed: int) -> torch.Tensor:
    """Sabit noktasiz tohumlu permutasyon (karistirilmis guven kolu)."""
    g = torch.Generator().manual_seed(int(seed))
    while True:
        p = torch.randperm(n, generator=g)
        if not bool((p == torch.arange(n)).any()):
            return p


GLM_TEMP_RANGE = (0.25, 4.0)   # kullanici karari 2026-11-19 (M5 §4.3): [0,5; 2] gonderici sirasini az bozuyordu (Spearman ~0,88)


def sender_temperatures(n_agents: int, seed: int, lo: float = GLM_TEMP_RANGE[0], hi: float = GLM_TEMP_RANGE[1]) -> list:
    """GLM kolu: gonderici basina T_j ~ log-tekduze [lo, hi], numpy PCG64 tohumlu (surecler arasi deterministik)."""
    if not (0 < lo < hi):
        raise ValueError(f"0 < lo < hi olmali: {lo}, {hi}")
    rng = np.random.default_rng(int(seed))
    return [float(t) for t in np.exp(rng.uniform(np.log(lo), np.log(hi), int(n_agents)))]


def conf_scale(p_up: np.ndarray, ok: np.ndarray, temp: Optional[Sequence[float]] = None) -> float:
    """zlogodds z_scale = std(|T_j logit p|) tahmini olan hucrelerde. p_up [T, N] (ham p_up_cal), ok [T, N]; satir
    secimi (egitim penceresi) cagirana aittir. temp verilirse bozulmus terimin std'si (GLM)."""
    p = np.clip(np.asarray(p_up, dtype=np.float64), 1e-6, 1 - 1e-6)
    lz = np.abs(np.log(p / (1 - p)))
    if temp is not None:
        lz = lz * np.asarray(temp, dtype=np.float64)[None, :]
    v = lz[np.asarray(ok, dtype=bool) & np.isfinite(lz)]
    if v.size < 2:
        raise ValueError("z_scale icin en az iki tahminli hucre gerekli")
    return float(v.std(ddof=1))


def confidence_term(p_minus_half: torch.Tensor, ok: torch.Tensor, mode: str, c_min: float, c_max: float,
                    z_scale: float) -> torch.Tensor:
    """Gonderici guven terimi b_j [..., N] (alicidan bagimsiz). p_minus_half = p_up_cal - 0,5 (gozlem sutunu)."""
    if mode == 'prob':
        c = torch.clamp(0.5 + p_minus_half.abs(), c_min, c_max)
        b = torch.log(c)
        floor = torch.log(torch.as_tensor(c_min, dtype=b.dtype, device=b.device))
    elif mode == 'zlogodds':
        p = torch.clamp(0.5 + p_minus_half, 1e-6, 1 - 1e-6)
        b = torch.log(p / (1 - p)).abs() / z_scale
        floor = torch.zeros((), dtype=b.dtype, device=b.device)
    else:
        raise ValueError(f"conf_mode {CONF_MODES[1:]}: {mode}")
    return torch.where(ok > 0.5, b, floor)


class CommPolicy(Model):
    """Kodlayici -> `rounds` tur (dikkat + artik + LayerNorm) -> [kendi kod || mesaj sonrasi] -> cikti."""

    def __init__(self, kind: str, hidden: int, heads: int, rounds: int, edges: Optional[Sequence] = None,
                 conf_mode: str = 'none', conf_index: Optional[int] = None, conf_ok_index: Optional[int] = None,
                 beta_mode: str = 'fixed', beta_init: float = 0.0, gate: bool = False, gate_b_init: float = 4.6,
                 c_min: float = 0.5, c_max: float = 0.99, z_scale: float = 1.0, shuffle_conf: bool = False,
                 shuffle_seed: int = 0, conf_temp: Optional[Sequence[float]] = None, **kwargs):
        super().__init__(
            input_spec=kwargs.pop('input_spec'), output_spec=kwargs.pop('output_spec'),
            agent_group=kwargs.pop('agent_group'), input_has_agent_dim=kwargs.pop('input_has_agent_dim'),
            n_agents=kwargs.pop('n_agents'), centralised=kwargs.pop('centralised'),
            share_params=kwargs.pop('share_params'), device=kwargs.pop('device'),
            action_spec=kwargs.pop('action_spec'), model_index=kwargs.pop('model_index'),
            is_critic=kwargs.pop('is_critic'),
        )
        if kind not in KINDS:
            raise ValueError(f"kind {KINDS}: {kind}")
        if not (self.input_has_agent_dim and self.share_params) or self.centralised or self.is_critic:
            raise ValueError('CommPolicy yalniz ajan eksenli, tam paylasimli, merkezi olmayan politika (K24)')
        if kind == 'gat' and not edges:
            raise ValueError("gat kolu kenar listesi ister")
        if kind in ('attn', 'self') and edges:
            raise ValueError(f"{kind} kolu kenar listesi almaz")
        if kind == 'self' and conf_mode != 'none':
            raise ValueError("self kolu iletisimsizdir; guven terimi anlamsiz")
        if hidden % heads:
            raise ValueError(f"hidden ({hidden}) heads'e ({heads}) bolunmeli")
        if conf_mode not in CONF_MODES or beta_mode not in BETA_MODES:
            raise ValueError(f"conf_mode {CONF_MODES}, beta_mode {BETA_MODES}: {conf_mode}, {beta_mode}")
        uses_conf = conf_mode != 'none'
        if not uses_conf and (gate or shuffle_conf or beta_mode == 'learned' or beta_init or conf_temp is not None):
            raise ValueError("beta / gecit / karistirma / sicaklik yalniz conf_mode != 'none' ile")
        if conf_temp is not None and (shuffle_conf or len(conf_temp) != self.n_agents
                                      or not all(float(t) > 0 for t in conf_temp)):
            raise ValueError("conf_temp: n_agents uzunlukta pozitif sicaklik; karistirilmis guvenle birlikte kullanilmaz")
        n_in = sum(s.shape[-1] for s in self.input_spec.values(True, True))
        if uses_conf and not (conf_index is not None and conf_ok_index is not None
                              and 0 <= conf_index < n_in and 0 <= conf_ok_index < n_in):
            raise ValueError("conf_mode guven sutunu ister: conf_index, conf_ok_index (gozlem icinde)")
        if uses_conf and not (0.5 <= c_min < c_max <= 1.0 and z_scale > 0):
            raise ValueError("0,5 <= c_min < c_max <= 1 ve z_scale > 0 olmali")
        self.kind, self.heads, self.conf_mode = kind, heads, conf_mode
        self.conf_index, self.conf_ok_index = conf_index, conf_ok_index
        self.c_min, self.c_max, self.z_scale = float(c_min), float(c_max), float(z_scale)
        n_out = self.output_leaf_spec.shape[-1]
        self.enc = nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh()).to(self.device)
        self.attn = nn.ModuleList(nn.MultiheadAttention(hidden, heads, batch_first=True, device=self.device)
                                  for _ in range(rounds))
        self.norm = nn.ModuleList(nn.LayerNorm(hidden, device=self.device) for _ in range(rounds))
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, n_out)).to(self.device)
        if kind == 'gat':
            mask = graph_mask(self.n_agents, edges)
        elif kind == 'self':
            mask = graph_mask(self.n_agents, [])      # yalniz kendine dikkat: iletisim yok, katmanlar ayni
        else:
            mask = None
        self.register_buffer('attn_mask', None if mask is None else mask.to(self.device), persistent=False)
        if uses_conf:
            beta = torch.tensor(float(beta_init), device=self.device)
            if beta_mode == 'learned':
                self.beta = nn.Parameter(beta)
            else:
                self.register_buffer('beta', beta, persistent=False)
            self.gate = bool(gate)
            if self.gate:
                self.gate_a = nn.Parameter(torch.zeros((), device=self.device))
                self.gate_b = nn.Parameter(torch.tensor(float(gate_b_init), device=self.device))
            perm = derangement(self.n_agents, shuffle_seed) if shuffle_conf else None
            self.register_buffer('conf_perm', None if perm is None else perm.to(self.device), persistent=False)
            temp = None if conf_temp is None else torch.tensor([float(t) for t in conf_temp], device=self.device)
            self.register_buffer('conf_temp', temp, persistent=False)
        # M7a karsi-olgusal mesaj maskeleme (yalniz degerlendirme; kaydedilmez): None iken ileri gecis degismez (bit-es).
        self.register_buffer('sender_block', None, persistent=False)

    def set_sender_mask(self, senders=None) -> None:
        """Verilen gondericilerin mesajini diger alicilara kapat (alici kendi kodunu gormeye devam eder). None = kapali.

        Yalniz degerlendirme araci (M7a): egitimde kullanilmaz, durum sozlugune yazilmaz. `self` kolunda zaten
        iletisim olmadigi icin etkisizdir."""
        if senders is None:
            self.sender_block = None
            return
        idx = sorted({int(j) for j in senders})
        if any(j < 0 or j >= self.n_agents for j in idx):
            raise ValueError(f"gonderici 0..{self.n_agents - 1} araliginda olmali: {idx}")
        blk = torch.zeros(self.n_agents, self.n_agents, dtype=torch.bool, device=self.device)
        if idx:
            blk[:, idx] = True
            blk[idx, idx] = False      # alici j kendine dikkati korur (tum anahtarlar maskeli satir olmaz)
        self.sender_block = blk

    def _bool_mask(self):
        """Graf maskesi OR gonderici maskesi; ikisi de yoksa None."""
        if self.sender_block is None:
            return self.attn_mask
        return self.sender_block if self.attn_mask is None else (self.attn_mask | self.sender_block)

    def conf_terms(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, N, F] -> gonderici guven terimi b [B, N] (karistirma ya da gonderici sicakligi dahil)."""
        pm = x[..., self.conf_index]
        if self.conf_temp is not None:     # GLM: yalniz guven teriminde p_j <- sigmoid(T_j logit p_j)
            p = torch.clamp(0.5 + pm, 1e-6, 1 - 1e-6)
            pm = torch.sigmoid(self.conf_temp * torch.log(p / (1 - p))) - 0.5
        b = confidence_term(pm, x[..., self.conf_ok_index], self.conf_mode, self.c_min, self.c_max, self.z_scale)
        return b if self.conf_perm is None else b[..., self.conf_perm]

    def _score_mask(self, b: torch.Tensor) -> torch.Tensor:
        """Ek puan maskesi [B*heads, N, N]: beta * b_j (her alici satirinda ayni) + graf engeli (-inf)."""
        B, N = b.shape
        m = (self.beta * b)[:, None, :].expand(B, N, N)
        bm = self._bool_mask()
        if bm is not None:
            m = m.masked_fill(bm, float('-inf'))
        return m.repeat_interleave(self.heads, dim=0)

    def lack_of_confidence(self, x: torch.Tensor) -> torch.Tensor:
        """Alicinin kendi guvensizligi d_i [B, N] (karistirilmaz): prob 1 - c_i; zlogodds -|logit p_i| / z_scale."""
        own = confidence_term(x[..., self.conf_index], x[..., self.conf_ok_index], self.conf_mode,
                              self.c_min, self.c_max, self.z_scale)
        return 1.0 - torch.exp(own) if self.conf_mode == 'prob' else -own

    def _gate(self, x: torch.Tensor) -> torch.Tensor:
        """g_i = sigmoid(a * d_i + b) [B, N, 1]; a > 0 ise dusuk oz-guven mesajlara daha cok agirlik verir."""
        return torch.sigmoid(self.gate_a * self.lack_of_confidence(x) + self.gate_b)[..., None]

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        x = torch.cat([tensordict.get(k) for k in self.in_keys], dim=-1)  # [..., N, F]
        lead = x.shape[:-2]
        xf = x.reshape(-1, *x.shape[-2:])
        h = self.enc(xf)  # [B, N, H]
        own = h
        if self.conf_mode == 'none':
            mask, g = self._bool_mask(), None
        else:
            b = self.conf_terms(xf)
            mask = self._score_mask(b)
            g = self._gate(xf) if self.gate else None
        for att, ln in zip(self.attn, self.norm):
            m, _ = att(h, h, h, attn_mask=mask, need_weights=False)
            h = ln(h + (m if g is None else g * m))
        out = self.head(torch.cat([own, h], dim=-1))
        tensordict.set(self.out_key, out.reshape(*lead, *out.shape[-2:]))
        return tensordict


@dataclass
class CommPolicyConfig(ModelConfig):
    kind: str = 'attn'
    hidden: int = 112
    heads: int = 4
    rounds: int = 1
    edges: Optional[list] = field(default=None)
    conf_mode: str = 'none'
    conf_index: Optional[int] = None
    conf_ok_index: Optional[int] = None
    beta_mode: str = 'fixed'
    beta_init: float = 0.0
    gate: bool = False
    gate_b_init: float = 4.6
    c_min: float = 0.5
    c_max: float = 0.99
    z_scale: float = 1.0
    shuffle_conf: bool = False
    shuffle_seed: int = 0
    conf_temp: Optional[list] = field(default=None)

    @staticmethod
    def associated_class():
        return CommPolicy


def comm_param_count(n_in: int, n_out: int, hidden: int, rounds: int = 1, beta_learned: bool = False,
                     gate: bool = False) -> int:
    """CommPolicy parametre sayisi (attn ve gat ayni): kodlayici + rounds x (MHA 4H^2+4H + LN 2H) + cikti
    (+1 ogrenilen beta, +2 gecit a, b)."""
    H = hidden
    return ((n_in * H + H) + rounds * (4 * H * H + 4 * H + 2 * H) + (2 * H * H + H) + (H * n_out + n_out)
            + int(beta_learned) + 2 * int(gate))


def mlp_param_count(n_in: int, n_out: int, cells: Sequence[int] = (256, 256)) -> int:
    """BenchMARL MLP (ajan basina, paylasimli) parametre sayisi."""
    dims = [n_in, *cells, n_out]
    return sum(a * b + b for a, b in zip(dims[:-1], dims[1:]))
