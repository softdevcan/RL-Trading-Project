/**
 * Prediction Manager
 * Fiyat tahmini ve altın fiyatları için UI yöneticisi
 */

const PRED_DEBUG = false;
function predLog(...args) { if (PRED_DEBUG) console.log('[Prediction]', ...args); }
function predError(...args) { console.error('[Prediction]', ...args); }

class PredictionManager {
    constructor() {
        this.predVsActualChart = null;
        this.accuracyTrendChart = null;
        this.goldPriceChart = null;
        this.currentSymbol = null;
        this.currentHorizon = 'daily';
        this.init();
    }

    init() {
        predLog('Initializing...');
        this.setupEventListeners();
        this.loadSymbols();
        this.loadGoldPrices();
    }

    setupEventListeners() {
        const trainBtn = document.getElementById('pred-train-btn');
        if (trainBtn) trainBtn.addEventListener('click', () => this.trainModel());

        const predictBtn = document.getElementById('pred-predict-btn');
        if (predictBtn) predictBtn.addEventListener('click', () => this.predict());

        const evalBtn = document.getElementById('pred-eval-btn');
        if (evalBtn) evalBtn.addEventListener('click', () => this.evaluatePending());

        const refreshGoldBtn = document.getElementById('gold-refresh-btn');
        if (refreshGoldBtn) refreshGoldBtn.addEventListener('click', () => this.loadGoldPrices());

        const symbolSel = document.getElementById('pred-symbol');
        if (symbolSel) symbolSel.addEventListener('change', (e) => {
            this.currentSymbol = e.target.value;
            this.loadPerformance();
            this.loadPredictionHistory();
        });

        const horizonRadios = document.querySelectorAll('input[name="pred-horizon"]');
        horizonRadios.forEach(r => r.addEventListener('change', (e) => {
            this.currentHorizon = e.target.value;
            this.loadPerformance();
            this.loadPredictionHistory();
        }));
    }

    // ------------------------------------------------------------------
    // Sembol Yükleme
    // ------------------------------------------------------------------

    async loadSymbols() {
        try {
            const resp = await fetch('/api/prediction/symbols');
            if (!resp.ok) throw new Error(await resp.text());
            const data = await resp.json();

            const sel = document.getElementById('pred-symbol');
            if (!sel) return;

            sel.innerHTML = '';

            const groups = [
                { label: 'BIST-30', symbols: data.bist30 || [] },
                { label: 'Altin & Doviz', symbols: [...(data.gold || []), ...(data.fx || []), ...(data.synthetic || [])] },
            ];

            groups.forEach(g => {
                if (g.symbols.length === 0) return;
                const optgroup = document.createElement('optgroup');
                optgroup.label = g.label;
                g.symbols.forEach(s => {
                    const opt = document.createElement('option');
                    opt.value = s;
                    opt.textContent = s;
                    optgroup.appendChild(opt);
                });
                sel.appendChild(optgroup);
            });

            if (sel.options.length > 0) {
                this.currentSymbol = sel.value;
            }

            predLog('Semboller yüklendi:', data);
        } catch (err) {
            predError('Semboller yüklenemedi:', err);
        }
    }

    // ------------------------------------------------------------------
    // Model Eğitimi
    // ------------------------------------------------------------------

    async trainModel() {
        const symbol = document.getElementById('pred-symbol')?.value;
        const horizon = this.currentHorizon;
        const startDate = document.getElementById('pred-start-date')?.value || '2018-01-01';

        if (!symbol) {
            this._showError('pred-error', 'Sembol seçin');
            return;
        }

        const btn = document.getElementById('pred-train-btn');
        this._setLoading(btn, true, 'Egitiliyor...');
        this._clearError('pred-error');

        try {
            const resp = await fetch('/api/prediction/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ symbol, horizon, start_date: startDate, test_ratio: 0.2 }),
            });

            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || 'Egitim basarisiz');
            }

            const result = await resp.json();
            this._renderTrainResult(result);
            predLog('Egitim sonucu:', result);

        } catch (err) {
            predError('Egitim hatasi:', err);
            this._showError('pred-error', err.message);
        } finally {
            this._setLoading(btn, false, 'Model Egit');
        }
    }

    _renderTrainResult(result) {
        const container = document.getElementById('pred-train-result');
        if (!container) return;

        const testM = result.test_metrics || {};
        container.innerHTML = `
            <div class="train-result-card">
                <h4>${result.symbol} - ${result.horizon === 'daily' ? 'Gunluk' : 'Haftalik'} Model Egitildi</h4>
                <div class="metrics-row">
                    <div class="metric-item">
                        <span class="metric-label">Test MAPE</span>
                        <span class="metric-val">${(testM.mape || 0).toFixed(2)}%</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Test RMSE</span>
                        <span class="metric-val">${(testM.rmse || 0).toFixed(4)}</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Yon Dogrulugu</span>
                        <span class="metric-val">${(testM.direction_accuracy || 0).toFixed(1)}%</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Egitim Ornegi</span>
                        <span class="metric-val">${result.n_train}</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Test Ornegi</span>
                        <span class="metric-val">${result.n_test}</span>
                    </div>
                </div>
                <p class="result-path">Model: ${result.model_path}</p>
            </div>
        `;
        container.style.display = 'block';
    }

    // ------------------------------------------------------------------
    // Tahmin
    // ------------------------------------------------------------------

    async predict() {
        const symbol = document.getElementById('pred-symbol')?.value;
        const horizon = this.currentHorizon;

        if (!symbol) {
            this._showError('pred-error', 'Sembol secin');
            return;
        }

        const btn = document.getElementById('pred-predict-btn');
        this._setLoading(btn, true, 'Tahmin yapiliyor...');
        this._clearError('pred-error');

        try {
            const resp = await fetch('/api/prediction/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ symbols: [symbol], horizon }),
            });

            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || 'Tahmin basarisiz');
            }

            const data = await resp.json();
            if (data.predictions && data.predictions.length > 0) {
                this._renderPredictionCard(data.predictions[0]);
            } else {
                this._showError('pred-error', 'Tahmin uretilmedi. Once modeli egitin.');
            }

        } catch (err) {
            predError('Tahmin hatasi:', err);
            this._showError('pred-error', err.message);
        } finally {
            this._setLoading(btn, false, 'Tahmin Yap');
        }
    }

    _renderPredictionCard(pred) {
        const container = document.getElementById('pred-result-card');
        if (!container) return;

        const isUp = pred.predicted_direction === 'UP';
        const dirClass = isUp ? 'direction-up' : 'direction-down';
        const arrow = isUp ? '▲' : '▼';
        const changePct = pred.predicted_change_pct || 0;
        const confidence = Math.round((pred.confidence || 0.65) * 100);

        container.innerHTML = `
            <div class="prediction-card">
                <div class="pred-header">
                    <span class="pred-symbol">${pred.symbol}</span>
                    <span class="pred-horizon-badge">${pred.horizon === 'daily' ? 'Gunluk' : 'Haftalik'}</span>
                </div>
                <div class="pred-main">
                    <div class="pred-price">
                        <span class="pred-price-label">Tahmin Edilen Kapanış</span>
                        <span class="pred-price-value">${pred.predicted_close?.toFixed(2)}</span>
                    </div>
                    <div class="pred-direction ${dirClass}">
                        <span class="direction-arrow">${arrow}</span>
                        <span class="direction-pct">${Math.abs(changePct).toFixed(2)}%</span>
                    </div>
                </div>
                <div class="pred-meta">
                    <span>Mevcut: ${pred.current_close?.toFixed(2)}</span>
                    <span>Tarih: ${pred.prediction_date}</span>
                </div>
                <div class="confidence-section">
                    <span class="confidence-label">Guven: ${confidence}%</span>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width:${confidence}%"></div>
                    </div>
                </div>
            </div>
        `;
        container.style.display = 'block';
    }

    // ------------------------------------------------------------------
    // Bekleyen Değerlendirme
    // ------------------------------------------------------------------

    async evaluatePending() {
        const btn = document.getElementById('pred-eval-btn');
        this._setLoading(btn, true, 'Degerlendirilyor...');

        try {
            const resp = await fetch('/api/prediction/evaluate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(null),
            });
            const data = await resp.json();
            alert(`${data.evaluated_count} tahmin degerlendirildi`);
            this.loadPerformance();
            this.loadPredictionHistory();
        } catch (err) {
            predError('Degerlendirme hatasi:', err);
        } finally {
            this._setLoading(btn, false, 'Bekleyenleri Degerlendir');
        }
    }

    // ------------------------------------------------------------------
    // Performans Metrikleri
    // ------------------------------------------------------------------

    async loadPerformance() {
        const symbol = this.currentSymbol || document.getElementById('pred-symbol')?.value;
        if (!symbol) return;

        try {
            const resp = await fetch(
                `/api/prediction/performance/${encodeURIComponent(symbol)}?horizon=${this.currentHorizon}`
            );
            if (!resp.ok) return;
            const data = await resp.json();
            this._renderPerformanceCard(data);
        } catch (err) {
            predError('Performans yuklenemedi:', err);
        }
    }

    _renderPerformanceCard(metrics) {
        const el = document.getElementById('pred-performance-card');
        if (!el) return;

        if (!metrics.n_predictions || metrics.n_predictions === 0) {
            el.innerHTML = '<p class="info-text">Henuz degerlendirilen tahmin yok.</p>';
            return;
        }

        const mape = metrics.mape != null ? metrics.mape.toFixed(2) + '%' : '-';
        const rmse = metrics.rmse != null ? metrics.rmse.toFixed(4) : '-';
        const mae = metrics.mae != null ? metrics.mae.toFixed(4) : '-';
        const dirAcc = metrics.direction_accuracy != null
            ? metrics.direction_accuracy.toFixed(1) + '%' : '-';
        const dirClass = (metrics.direction_accuracy || 0) >= 55 ? 'good' : 'bad';

        el.innerHTML = `
            <div class="metrics-row">
                <div class="metric-item">
                    <span class="metric-label">MAPE</span>
                    <span class="metric-val">${mape}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">RMSE</span>
                    <span class="metric-val">${rmse}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">MAE</span>
                    <span class="metric-val">${mae}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Yon Dogrulugu</span>
                    <span class="metric-val ${dirClass}">${dirAcc}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Tahmin Sayisi</span>
                    <span class="metric-val">${metrics.n_predictions}</span>
                </div>
            </div>
        `;
    }

    // ------------------------------------------------------------------
    // Tahmin Geçmişi Tablosu
    // ------------------------------------------------------------------

    async loadPredictionHistory() {
        const symbol = this.currentSymbol || document.getElementById('pred-symbol')?.value;
        if (!symbol) return;

        try {
            const resp = await fetch(
                `/api/prediction/predictions/${encodeURIComponent(symbol)}?horizon=${this.currentHorizon}`
            );
            if (!resp.ok) return;
            const data = await resp.json();

            this._renderHistoryTable(data.history || []);
            this._loadChartData(symbol);

        } catch (err) {
            predError('Gecmis yuklenemedi:', err);
        }
    }

    _renderHistoryTable(history) {
        const tbody = document.getElementById('pred-history-tbody');
        if (!tbody) return;

        if (history.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" style="text-align:center">Henuz tahmin yok</td></tr>';
            return;
        }

        // Son 30 tahmini ters sırada göster
        const recent = [...history].reverse().slice(0, 30);
        tbody.innerHTML = recent.map(row => {
            const actual = row.actual_close != null ? row.actual_close.toFixed(2) : '-';
            const error = row.error_pct != null ? row.error_pct.toFixed(2) + '%' : '-';
            const dirOk = row.direction_correct == null ? '-'
                : row.direction_correct ? '<span class="badge-ok">✓</span>'
                    : '<span class="badge-err">✗</span>';
            const dirArrow = row.predicted_direction === 'UP' ? '▲' : '▼';
            const dirClass = row.predicted_direction === 'UP' ? 'direction-up' : 'direction-down';

            return `<tr>
                <td>${row.prediction_date}</td>
                <td>${row.predicted_close?.toFixed(2)}</td>
                <td class="${dirClass}">${dirArrow} ${Math.abs(row.predicted_change_pct || 0).toFixed(2)}%</td>
                <td>${actual}</td>
                <td>${error}</td>
                <td>${dirOk}</td>
                <td>${Math.round((row.confidence || 0) * 100)}%</td>
            </tr>`;
        }).join('');
    }

    // ------------------------------------------------------------------
    // Grafikler
    // ------------------------------------------------------------------

    async _loadChartData(symbol) {
        try {
            const resp = await fetch(
                `/api/prediction/chart-data/${encodeURIComponent(symbol)}?horizon=${this.currentHorizon}&days=60`
            );
            if (!resp.ok) return;
            const data = await resp.json();
            this._renderPredVsActualChart(data.chart || {});
            this._renderAccuracyTrendChart(data.trend || {});
        } catch (err) {
            predError('Chart verisi yuklenemedi:', err);
        }
    }

    _renderPredVsActualChart(chartData) {
        const canvas = document.getElementById('pred-vs-actual-chart');
        if (!canvas) return;

        if (this.predVsActualChart) {
            this.predVsActualChart.destroy();
        }

        const { dates = [], predicted = [], actual = [] } = chartData;
        if (dates.length === 0) return;

        this.predVsActualChart = new Chart(canvas, {
            type: 'line',
            data: {
                labels: dates,
                datasets: [
                    {
                        label: 'Tahmin',
                        data: predicted,
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59,130,246,0.1)',
                        borderWidth: 2,
                        pointRadius: 3,
                        tension: 0.3,
                    },
                    {
                        label: 'Gercek',
                        data: actual,
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16,185,129,0.1)',
                        borderWidth: 2,
                        pointRadius: 3,
                        tension: 0.3,
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { labels: { color: '#94a3b8' } } },
                scales: {
                    x: { ticks: { color: '#64748b', maxTicksLimit: 8 }, grid: { color: '#1e293b' } },
                    y: { ticks: { color: '#64748b' }, grid: { color: '#1e293b' } },
                }
            }
        });
    }

    _renderAccuracyTrendChart(trendData) {
        const canvas = document.getElementById('pred-accuracy-trend-chart');
        if (!canvas) return;

        if (this.accuracyTrendChart) {
            this.accuracyTrendChart.destroy();
        }

        const { dates = [], mape = [], direction_accuracy = [] } = trendData;
        if (dates.length === 0) return;

        this.accuracyTrendChart = new Chart(canvas, {
            type: 'line',
            data: {
                labels: dates,
                datasets: [
                    {
                        label: 'MAPE (%)',
                        data: mape,
                        borderColor: '#f59e0b',
                        backgroundColor: 'rgba(245,158,11,0.1)',
                        borderWidth: 2,
                        yAxisID: 'y1',
                        tension: 0.3,
                    },
                    {
                        label: 'Yon Dogrulugu (%)',
                        data: direction_accuracy,
                        borderColor: '#a855f7',
                        backgroundColor: 'rgba(168,85,247,0.1)',
                        borderWidth: 2,
                        yAxisID: 'y2',
                        tension: 0.3,
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { labels: { color: '#94a3b8' } } },
                scales: {
                    x: { ticks: { color: '#64748b', maxTicksLimit: 8 }, grid: { color: '#1e293b' } },
                    y1: {
                        type: 'linear', position: 'left',
                        ticks: { color: '#f59e0b' }, grid: { color: '#1e293b' },
                        title: { display: true, text: 'MAPE%', color: '#f59e0b' }
                    },
                    y2: {
                        type: 'linear', position: 'right',
                        ticks: { color: '#a855f7' },
                        title: { display: true, text: 'Dir Acc%', color: '#a855f7' }
                    },
                }
            }
        });
    }

    // ------------------------------------------------------------------
    // Altın Fiyatları
    // ------------------------------------------------------------------

    async loadGoldPrices() {
        const container = document.getElementById('gold-prices-container');
        if (container) container.innerHTML = '<p class="info-text">Yukluyor...</p>';

        try {
            const resp = await fetch('/api/prediction/gold/prices');
            if (!resp.ok) throw new Error(await resp.text());
            const data = await resp.json();
            this._renderGoldPrices(data.prices || []);
            this._loadGoldHistoryChart();
        } catch (err) {
            predError('Altin fiyatlari yuklenemedi:', err);
            if (container) container.innerHTML = '<p class="info-text">Altin verileri alinirken hata olustu.</p>';
        }
    }

    _renderGoldPrices(prices) {
        const container = document.getElementById('gold-prices-container');
        if (!container) return;

        if (prices.length === 0) {
            container.innerHTML = '<p class="info-text">Altin verisi alinamadi.</p>';
            return;
        }

        container.innerHTML = prices.map(p => {
            const isUp = p.change_pct >= 0;
            const dirClass = isUp ? 'direction-up' : 'direction-down';
            const arrow = isUp ? '▲' : '▼';
            return `
                <div class="gold-price-card">
                    <div class="gold-name">${p.name}</div>
                    <div class="gold-price">${p.close.toFixed(2)} <span class="gold-currency">${p.currency}</span></div>
                    <div class="gold-change ${dirClass}">${arrow} ${Math.abs(p.change_pct).toFixed(2)}%</div>
                    <div class="gold-date">${p.date}</div>
                </div>
            `;
        }).join('');
    }

    async _loadGoldHistoryChart() {
        const goldSymSel = document.getElementById('gold-symbol-select');
        const sym = goldSymSel?.value || 'GOLD_GRAM_TRY';

        try {
            const resp = await fetch(
                `/api/prediction/gold/history?symbol=${encodeURIComponent(sym)}&days=90`
            );
            if (!resp.ok) return;
            const data = await resp.json();
            this._renderGoldChart(data);
        } catch (err) {
            predError('Altin gecmisi yuklenemedi:', err);
        }
    }

    _renderGoldChart(data) {
        const canvas = document.getElementById('gold-price-chart');
        if (!canvas) return;

        if (this.goldPriceChart) {
            this.goldPriceChart.destroy();
        }

        this.goldPriceChart = new Chart(canvas, {
            type: 'line',
            data: {
                labels: data.dates || [],
                datasets: [{
                    label: data.symbol,
                    data: data.close || [],
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245,158,11,0.15)',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: true,
                    tension: 0.3,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { labels: { color: '#94a3b8' } } },
                scales: {
                    x: { ticks: { color: '#64748b', maxTicksLimit: 8 }, grid: { color: '#1e293b' } },
                    y: { ticks: { color: '#64748b' }, grid: { color: '#1e293b' } },
                }
            }
        });
    }

    // ------------------------------------------------------------------
    // Yardımcı
    // ------------------------------------------------------------------

    _setLoading(btn, loading, text) {
        if (!btn) return;
        btn.disabled = loading;
        btn.textContent = text;
    }

    _showError(containerId, msg) {
        const el = document.getElementById(containerId);
        if (el) {
            el.innerHTML = `<div class="error-message">${msg}</div>`;
            el.style.display = 'block';
        }
    }

    _clearError(containerId) {
        const el = document.getElementById(containerId);
        if (el) el.innerHTML = '';
    }
}

// Tab değişiminde chart yeniden render
function onPredictionTabActivated() {
    if (window.predictionManager) {
        if (window.predictionManager.predVsActualChart) {
            window.predictionManager.predVsActualChart.resize();
        }
        if (window.predictionManager.accuracyTrendChart) {
            window.predictionManager.accuracyTrendChart.resize();
        }
        if (window.predictionManager.goldPriceChart) {
            window.predictionManager.goldPriceChart.resize();
        }
    }
}
