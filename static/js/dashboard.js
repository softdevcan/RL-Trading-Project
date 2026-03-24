/**
 * RL Trading Dashboard - Main JavaScript
 * Handles all dashboard functionality, API calls, and chart updates
 */

function escapeHtml(str) {
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

// Global state
let isTraining = false;
let statusCheckInterval = null;
let selectedModel = null;
let performanceChart = null;
let trainingProgressChart = null;
let algorithmComparisonChart = null;
let allModels = [];

/**
 * Tab Management
 */
function switchTab(tabName, event) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });

    // Show selected tab
    document.getElementById(tabName).classList.add('active');
    if (event && event.target) {
        event.target.classList.add('active');
    }

    // Load data for specific tabs
    if (tabName === 'models') {
        loadModelsComparison();
    } else if (tabName === 'daily-trading') {
        // Reload models when switching to daily trading tab
        if (window.dailyTradingManager) {
            console.log('Switching to daily-trading tab, reloading models...');
            window.dailyTradingManager.loadModels();
        }
    } else if (tabName === 'academic') {
        // Reload analysis when switching to academic tab
        if (window.academicAnalysisManager) {
            console.log('Switching to academic tab, reloading analysis...');
            window.academicAnalysisManager.loadExistingResults();
        }
    }
}

/**
 * Chart Initialization
 */
function initCharts() {
    // Performance Chart
    const perfCtx = document.getElementById('performanceChart');
    performanceChart = new Chart(perfCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Portfolio Value (₺)',
                data: [],
                borderColor: '#667eea',
                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    ticks: {
                        callback: function(value) {
                            return '₺' + value.toLocaleString('tr-TR');
                        }
                    }
                }
            }
        }
    });

    // Training Progress Chart - Removed (training is synchronous)

    // Algorithm Comparison Chart
    const algoCtx = document.getElementById('algorithmComparisonChart');
    algorithmComparisonChart = new Chart(algoCtx, {
        type: 'bar',
        data: {
            labels: ['A2C', 'PPO', 'TD3'],
            datasets: [{
                label: 'Average Sharpe Ratio',
                data: [0, 0, 0],
                backgroundColor: [
                    'rgba(102, 126, 234, 0.8)',
                    'rgba(79, 172, 254, 0.8)',
                    'rgba(67, 233, 123, 0.8)'
                ],
                borderColor: [
                    '#667eea',
                    '#4facfe',
                    '#43e97b'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

/**
 * Training Form Submission
 */
function initTrainingForm() {
    document.getElementById('trainingForm').addEventListener('submit', async (e) => {
        e.preventDefault();

        if (isTraining) {
            alert('Eğitim zaten devam ediyor!');
            return;
        }

        const formData = {
            algorithm: document.getElementById('algorithm').value,
            phase: parseInt(document.getElementById('phase').value),
            total_timesteps: parseInt(document.getElementById('timesteps').value),
            learning_rate: 0.0007,  // Default value (will be overridden by algorithm-specific optimal LR in backend)
            initial_balance: parseFloat(document.getElementById('initial_balance').value),
            commission_rate: 0.001,
            max_shares_per_trade: 100
        };

        // Add hyperparameter study if selected
        const hyperparamStudy = document.getElementById('hyperparameter_study').value;
        if (hyperparamStudy) {
            formData.hyperparameter_study = hyperparamStudy;
        }

        try {
            const response = await fetch('/api/trading/train', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });

            // Try to parse JSON response
            let data;
            try {
                data = await response.json();
            } catch (jsonError) {
                console.error('JSON parse error:', jsonError);
                showError('❌ Sunucu yanıt hatası. Lütfen tekrar deneyin.');
                return;
            }

            if (response.ok) {
                // Training started - set state and start polling
                isTraining = true;
                const trainButton = document.getElementById('trainButton');
                if (trainButton) {
                    trainButton.disabled = true;
                    trainButton.textContent = '⏳ Eğitim Devam Ediyor...';
                }

                const progressContainer = document.getElementById('progressContainer');
                if (progressContainer) {
                    progressContainer.style.display = 'block';
                }

                updateStatus('training');

                // Start polling for status updates
                startStatusCheck();

                // Show success message
                showError('✅ Eğitim başlatıldı! Bu işlem birkaç dakika sürebilir. Model eğitimi tamamlandığında bu sayfa otomatik olarak güncellenecek.', 'success');
            } else {
                showError('❌ ' + (data.detail || 'Eğitim başlatılamadı'));
            }
        } catch (error) {
            console.error('Training request error:', error);
            showError('❌ API bağlantı hatası: ' + error.message);
        }
    });
}

/**
 * Status Check
 */
async function checkStatus() {
    try {
        const response = await fetch('/api/trading/train/status');
        const data = await response.json();

        if (data.is_training) {
            const progress = (data.progress * 100).toFixed(1);

            const progressBar = document.getElementById('progressBar');
            if (progressBar) {
                progressBar.style.width = progress + '%';
                progressBar.textContent = progress + '%';
            }

            const stepInfo = document.getElementById('stepInfo');
            if (stepInfo) {
                stepInfo.textContent = `Step: ${data.current_step.toLocaleString()} / ${data.total_steps.toLocaleString()}`;
            }

            // Update training progress chart
            updateTrainingChart(data.current_step, data.progress);
        } else {
            if (isTraining) {
                // Training just finished
                isTraining = false;
                updateStatus('idle');

                const trainButton = document.getElementById('trainButton');
                if (trainButton) {
                    trainButton.disabled = false;
                    trainButton.textContent = '🚀 Eğitimi Başlat';
                }

                const progressContainer = document.getElementById('progressContainer');
                if (progressContainer) {
                    progressContainer.style.display = 'none';
                }

                if (data.error) {
                    showError('❌ Eğitim hatası: ' + data.error);
                } else if (data.metrics) {
                    showError('✅ Eğitim başarıyla tamamlandı! Model kaydedildi.', 'success');
                    updateMetrics(data.metrics);
                    loadModels();
                }

                stopStatusCheck();
            }
        }
    } catch (error) {
        console.error('Status check error:', error);
    }
}

function updateTrainingChart(step, progress) {
    // Training chart removed - training is synchronous
    return;
}

function startStatusCheck() {
    statusCheckInterval = setInterval(checkStatus, 2000);
}

function stopStatusCheck() {
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
        statusCheckInterval = null;
    }
}

function updateStatus(status) {
    const badge = document.getElementById('statusBadge');
    if (status === 'training') {
        badge.textContent = 'Eğitim Devam Ediyor...';
        badge.className = 'status-badge status-training';
    } else {
        badge.textContent = 'Hazır';
        badge.className = 'status-badge status-idle';
    }
}

/**
 * Metrics Update
 */
function updateMetrics(metrics) {
    if (!metrics || Object.keys(metrics).length === 0) {
        return;
    }

    document.getElementById('metricReturn').textContent =
        metrics.cumulative_return ? (metrics.cumulative_return * 100).toFixed(2) + '%' : '-';
    document.getElementById('metricSharpe').textContent =
        metrics.sharpe_ratio ? metrics.sharpe_ratio.toFixed(4) : '-';
    document.getElementById('metricDrawdown').textContent =
        metrics.max_drawdown ? (metrics.max_drawdown * 100).toFixed(2) + '%' : '-';
    document.getElementById('metricPortfolio').textContent =
        metrics.final_portfolio_value ? '₺' + metrics.final_portfolio_value.toLocaleString('tr-TR', {maximumFractionDigits: 0}) : '-';
    document.getElementById('metricTrades').textContent =
        metrics.total_trades || '-';

    // Update performance chart
    if (metrics.final_portfolio_value) {
        updatePerformanceChart(metrics);
    }
}

function updatePerformanceChart(metrics) {
    // Use actual portfolio history from model metrics
    if (metrics.portfolio_history && metrics.portfolio_history.length > 0) {
        // Real portfolio values from test run
        const portfolioValues = metrics.portfolio_history;
        const labels = Array.from({length: portfolioValues.length}, (_, i) => `Day ${i + 1}`);

        performanceChart.data.labels = labels;
        performanceChart.data.datasets[0].data = portfolioValues;
        performanceChart.data.datasets[0].label = `Portfolio Value - ${metrics.algorithm || 'Model'}`;
        performanceChart.update();
    } else {
        // Fallback: Show only initial and final value
        const initialValue = 1000000;
        const finalValue = metrics.final_portfolio_value || initialValue;

        performanceChart.data.labels = ['Start', 'End'];
        performanceChart.data.datasets[0].data = [initialValue, finalValue];
        performanceChart.data.datasets[0].label = 'Portfolio Value (Summary)';
        performanceChart.update();
    }
}

/**
 * Model Management
 */
async function loadModels() {
    try {
        const response = await fetch('/api/trading/models');
        const models = await response.json();
        allModels = models;

        const modelsList = document.getElementById('modelsList');

        if (models.length === 0) {
            modelsList.innerHTML = '<p class="info-text">Henüz eğitilmiş model yok</p>';
            return;
        }

        modelsList.innerHTML = models.map((model, idx) => `
            <div class="model-item" onclick="selectModel(${idx})">
                <div class="model-name">${escapeHtml(model.name)}</div>
                <div class="model-meta">
                    Oluşturulma: ${new Date(model.created_at).toLocaleString('tr-TR')}
                </div>
                ${model.metrics.cumulative_return ? `
                    <div class="model-meta">
                        Return: ${(model.metrics.cumulative_return * 100).toFixed(2)}% |
                        Sharpe: ${model.metrics.sharpe_ratio?.toFixed(4) || 'N/A'} |
                        Trades: ${model.metrics.total_trades || 0}
                    </div>
                ` : ''}
                <button onclick="event.stopPropagation(); showModelDetails('${escapeHtml(model.name)}')"
                        class="btn-secondary" style="margin-top: 10px; width: 100%;">
                    📊 Detayları Gör
                </button>
            </div>
        `).join('');

        // Auto-select the most recent model (first in list)
        if (models.length > 0) {
            selectModel(0);
        }
    } catch (error) {
        console.error('Model loading error:', error);
        document.getElementById('modelsList').innerHTML =
            '<p class="error">Modeller yüklenemedi</p>';
    }
}

function selectModel(idx) {
    selectedModel = allModels[idx];

    // Update active model info
    const activeModelInfo = document.getElementById('activeModelInfo');
    activeModelInfo.innerHTML = `
        <div style="padding: 10px;">
            <div style="font-weight: 600; margin-bottom: 10px;">${escapeHtml(selectedModel.name)}</div>
            <div style="font-size: 14px; color: #666;">
                Algoritma: ${escapeHtml(selectedModel.metrics.algorithm || 'N/A')}<br>
                Oluşturulma: ${new Date(selectedModel.created_at).toLocaleString('tr-TR')}
            </div>
        </div>
    `;

    // Update metrics
    if (selectedModel.metrics) {
        updateMetrics(selectedModel.metrics);
    }

    // Highlight selected model
    document.querySelectorAll('.model-item').forEach((item, i) => {
        if (i === idx) {
            item.classList.add('selected');
        } else {
            item.classList.remove('selected');
        }
    });
}

/**
 * Model Comparison
 */
async function loadModelsComparison() {
    try {
        const response = await fetch('/api/trading/models');
        const models = await response.json();

        const tbody = document.getElementById('modelsComparisonTable');

        if (models.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" style="text-align: center; padding: 20px;">Henüz model yok</td></tr>';
            return;
        }

        tbody.innerHTML = models.map(model => `
            <tr>
                <td>${escapeHtml(model.name)}</td>
                <td>${escapeHtml(model.metrics.algorithm || 'N/A')}</td>
                <td>${model.metrics.cumulative_return ? (model.metrics.cumulative_return * 100).toFixed(2) : '-'}</td>
                <td>${model.metrics.sharpe_ratio ? model.metrics.sharpe_ratio.toFixed(4) : '-'}</td>
                <td>${model.metrics.max_drawdown ? (model.metrics.max_drawdown * 100).toFixed(2) : '-'}</td>
                <td>${model.metrics.total_trades || '-'}</td>
                <td>${new Date(model.created_at).toLocaleDateString('tr-TR')}</td>
            </tr>
        `).join('');

        // Update comparison chart
        updateComparisonChart(models);
    } catch (error) {
        console.error('Comparison loading error:', error);
    }
}

function updateComparisonChart(models) {
    const algoData = { 'A2C': [], 'PPO': [], 'TD3': [] };

    models.forEach(model => {
        const algo = model.metrics.algorithm;
        const sharpe = model.metrics.sharpe_ratio;
        if (algo && sharpe !== undefined) {
            if (!algoData[algo]) algoData[algo] = [];
            algoData[algo].push(sharpe);
        }
    });

    const avgSharpe = {
        'A2C': algoData['A2C'].length ? algoData['A2C'].reduce((a, b) => a + b, 0) / algoData['A2C'].length : 0,
        'PPO': algoData['PPO'].length ? algoData['PPO'].reduce((a, b) => a + b, 0) / algoData['PPO'].length : 0,
        'TD3': algoData['TD3'].length ? algoData['TD3'].reduce((a, b) => a + b, 0) / algoData['TD3'].length : 0
    };

    algorithmComparisonChart.data.datasets[0].data = [avgSharpe['A2C'], avgSharpe['PPO'], avgSharpe['TD3']];
    algorithmComparisonChart.update();
}

/**
 * Error Handling
 */
function showError(message, type = 'error') {
    const errorContainer = document.getElementById('errorContainer');
    const className = type === 'success' ? 'success' : 'error';
    errorContainer.innerHTML = `<div class="${className}">${message}</div>`;
    setTimeout(() => {
        errorContainer.innerHTML = '';
    }, 5000);
}

/**
 * Initialization
 */
function init() {
    console.log('Initializing RL Trading Dashboard...');
    initCharts();
    initTrainingForm();
    loadModels();
    checkStatus();
    initDataForm();
    // Load hyperparameter studies for the default selected algorithm
    loadHyperparameterStudies();
    console.log('Dashboard initialized successfully!');
}

function initDataForm() {
    // Set end date to today
    const today = new Date();
    const formatDate = (date) => {
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const day = String(date.getDate()).padStart(2, '0');
        return `${year}-${month}-${day}`;
    };
    document.getElementById('dataEndDate').value = formatDate(today);
}

/**
 * Data Management Functions
 */
async function generateDataWithDates(event) {
    const btn = event.target;
    btn.disabled = true;
    btn.textContent = '🔄 Veri Oluşturuluyor...';

    try {
        // Get values from form
        const startDate = document.getElementById('dataStartDate').value;
        const endDate = document.getElementById('dataEndDate').value;
        const phase = document.getElementById('dataPhase').value;

        // Validate dates
        if (!startDate || !endDate) {
            alert('❌ Lütfen başlangıç ve bitiş tarihlerini seçin!');
            return;
        }

        if (new Date(startDate) >= new Date(endDate)) {
            alert('❌ Başlangıç tarihi bitiş tarihinden önce olmalı!');
            return;
        }

        // Build URL with parameters
        const url = `/api/trading/data/generate?phase=${phase}&start_date=${startDate}&end_date=${endDate}`;

        const response = await fetch(url, {
            method: 'POST'
        });

        if (!response.ok) {
            throw new Error('Veri oluşturma başarısız');
        }

        const data = await response.json();

        // Show success message
        alert(`✅ Veri başarıyla oluşturuldu!\n\n` +
              `Hisseler: ${data.symbols.join(', ')}\n` +
              `Toplam Satır: ${data.total_rows}\n` +
              `Train: ${data.train_rows} | Val: ${data.val_rows} | Test: ${data.test_rows}\n` +
              `Tarih Aralığı: ${data.date_range.start} - ${data.date_range.end}`);

        // Update UI
        updateDataStats(data);
        checkDataInfo();

    } catch (error) {
        alert('❌ Hata: ' + error.message);
    } finally {
        btn.disabled = false;
        btn.textContent = '🔄 Veri Oluştur';
    }
}

// Quick date range helpers
function setQuickRange(range) {
    const endDate = new Date();
    const startDate = new Date();

    switch(range) {
        case '1y':
            startDate.setFullYear(endDate.getFullYear() - 1);
            break;
        case '3y':
            startDate.setFullYear(endDate.getFullYear() - 3);
            break;
        case '5y':
            startDate.setFullYear(endDate.getFullYear() - 5);
            break;
        default:
            startDate.setFullYear(2018);
    }

    // Format dates as YYYY-MM-DD
    const formatDate = (date) => {
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const day = String(date.getDate()).padStart(2, '0');
        return `${year}-${month}-${day}`;
    };

    document.getElementById('dataStartDate').value = formatDate(startDate);
    document.getElementById('dataEndDate').value = formatDate(endDate);
}

// Legacy function - keep for compatibility
async function generateData(event) {
    generateDataWithDates(event);
}

async function checkDataInfo() {
    try {
        const response = await fetch('/api/trading/data/info');
        const data = await response.json();

        const infoDiv = document.getElementById('dataInfo');
        const infoContent = document.getElementById('dataInfoContent');

        if (data.status === 'no_data') {
            infoContent.innerHTML = `<p style="color: orange;">⚠️ ${data.message}</p>`;
        } else {
            infoContent.innerHTML = `
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
                    <div><strong>Toplam Satır:</strong> ${data.total_rows}</div>
                    <div><strong>Hisseler:</strong> ${data.symbols.join(', ')}</div>
                    <div><strong>Train Set:</strong> ${data.train_rows} satır</div>
                    <div><strong>Val Set:</strong> ${data.val_rows} satır</div>
                    <div><strong>Test Set:</strong> ${data.test_rows} satır</div>
                    <div><strong>Tarih:</strong> ${data.date_range.start} - ${data.date_range.end}</div>
                    <div><strong>Sütunlar:</strong> ${data.columns.length} adet</div>
                    <div><strong>Sütun Listesi:</strong> ${data.columns.join(', ')}</div>
                </div>
            `;
            updateDataStats(data);
        }

        infoDiv.style.display = 'block';

    } catch (error) {
        console.error('Data info fetch error:', error);
        alert('❌ Veri bilgisi alınamadı: ' + error.message);
    }
}

async function loadDatasetsList() {
    try {
        const response = await fetch('/api/trading/data/list');
        const data = await response.json();

        const container = document.getElementById('datasetsContainer');
        const tbody = document.getElementById('datasetsTableBody');

        if (data.count === 0) {
            tbody.innerHTML = '<tr><td colspan="7" style="text-align: center; color: #888;">Henüz veri seti yok</td></tr>';
            container.style.display = 'block';
            return;
        }

        tbody.innerHTML = data.datasets.map(dataset => {
            if (dataset.error) {
                return `
                    <tr>
                        <td>${dataset.filename}</td>
                        <td>${dataset.size_mb} MB</td>
                        <td colspan="5" style="color: #ef4444;">❌ Hata: ${dataset.error}</td>
                    </tr>
                `;
            }

            const dateRange = `${dataset.date_range.start} → ${dataset.date_range.end}`;
            const symbols = dataset.symbols.length <= 3
                ? dataset.symbols.join(', ')
                : `${dataset.symbols.slice(0, 3).join(', ')} +${dataset.symbols.length - 3}`;
            const splits = `${dataset.train_rows} / ${dataset.val_rows} / ${dataset.test_rows}`;
            const created = new Date(dataset.created_at).toLocaleString('tr-TR', {
                year: 'numeric',
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            });

            return `
                <tr>
                    <td><strong>${dataset.filename}</strong></td>
                    <td>${dataset.size_mb} MB</td>
                    <td>${dateRange}</td>
                    <td title="${dataset.symbols.join(', ')}">${symbols}</td>
                    <td>${dataset.total_rows.toLocaleString()}</td>
                    <td>${splits}</td>
                    <td>${created}</td>
                </tr>
            `;
        }).join('');

        container.style.display = 'block';

    } catch (error) {
        console.error('Datasets list error:', error);
        alert('❌ Veri setleri listelenemedi: ' + error.message);
    }
}

function updateDataStats(data) {
    if (data.symbols) {
        document.getElementById('totalSymbols').textContent = data.symbols.length;
    }
    if (data.date_range) {
        document.getElementById('dataPeriod').textContent =
            `${data.date_range.start} - ${data.date_range.end}`;
    }
    if (data.total_rows) {
        document.getElementById('totalRows').textContent = data.total_rows;
    }
}

// Modal functions
let portfolioChartInstance = null;
let tradeActivityChartInstance = null;

async function showModelDetails(modelName) {
    try {
        const response = await fetch(`/api/trading/models/${modelName}/metrics`);
        const data = await response.json();

        // Set modal title
        document.getElementById('modalModelName').textContent =
            `${data.algorithm} - ${modelName}`;

        // Render portfolio chart
        renderPortfolioChart(data.portfolio_history || []);

        // Render trades table
        renderTradesTable(data.trades || []);

        // Render trade activity chart
        renderTradeActivityChart(data.trades || []);

        // Show modal
        document.getElementById('modelDetailsModal').style.display = 'block';

    } catch (error) {
        console.error('Error loading model details:', error);
        alert('❌ Model detayları yüklenemedi: ' + error.message);
    }
}

function closeModelDetails() {
    document.getElementById('modelDetailsModal').style.display = 'none';

    // Destroy charts to prevent memory leaks
    if (portfolioChartInstance) {
        portfolioChartInstance.destroy();
        portfolioChartInstance = null;
    }
    if (tradeActivityChartInstance) {
        tradeActivityChartInstance.destroy();
        tradeActivityChartInstance = null;
    }
}

// Close modal when clicking outside
window.onclick = function(event) {
    const modal = document.getElementById('modelDetailsModal');
    if (event.target === modal) {
        closeModelDetails();
    }
}

function renderPortfolioChart(portfolioHistory) {
    const ctx = document.getElementById('portfolioChart').getContext('2d');

    if (portfolioChartInstance) {
        portfolioChartInstance.destroy();
    }

    const steps = portfolioHistory.map((_, i) => i);

    portfolioChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: steps,
            datasets: [{
                label: 'Portfolio Value (₺)',
                data: portfolioHistory,
                borderColor: '#10b981',
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#f8fafc' }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Trading Step',
                        color: '#94a3b8'
                    },
                    ticks: { color: '#94a3b8' },
                    grid: { color: '#334155' }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Portfolio Value (₺)',
                        color: '#94a3b8'
                    },
                    ticks: {
                        color: '#94a3b8',
                        callback: function(value) {
                            return '₺' + value.toLocaleString();
                        }
                    },
                    grid: { color: '#334155' }
                }
            }
        }
    });
}

function renderTradesTable(trades) {
    const tbody = document.getElementById('tradesTableBody');

    if (trades.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" style="text-align: center;">İşlem bulunamadı</td></tr>';
        return;
    }

    tbody.innerHTML = trades.map(trade => `
        <tr>
            <td>${trade.index}</td>
            <td>${trade.symbol}</td>
            <td style="color: ${trade.action === 'BUY' ? '#10b981' : '#ef4444'}">
                ${trade.action}
            </td>
            <td>${trade.shares}</td>
            <td>₺${trade.price.toFixed(2)}</td>
            <td>₺${trade.total_cost.toFixed(2)}</td>
        </tr>
    `).join('');
}

function renderTradeActivityChart(trades) {
    const ctx = document.getElementById('tradeActivityChart').getContext('2d');

    if (tradeActivityChartInstance) {
        tradeActivityChartInstance.destroy();
    }

    // Group trades by symbol
    const tradesBySymbol = {};
    trades.forEach(trade => {
        if (!tradesBySymbol[trade.symbol]) {
            tradesBySymbol[trade.symbol] = [];
        }
        tradesBySymbol[trade.symbol].push({
            x: trade.index,
            y: trade.action === 'BUY' ? trade.shares : -trade.shares
        });
    });

    const colors = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6'];
    const datasets = Object.keys(tradesBySymbol).map((symbol, index) => ({
        label: symbol,
        data: tradesBySymbol[symbol],
        backgroundColor: colors[index % colors.length],
        borderColor: colors[index % colors.length]
    }));

    tradeActivityChartInstance = new Chart(ctx, {
        type: 'scatter',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#f8fafc' }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const shares = Math.abs(context.parsed.y);
                            const action = context.parsed.y > 0 ? 'BUY' : 'SELL';
                            return `${context.dataset.label}: ${action} ${shares} shares`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Trade Index',
                        color: '#94a3b8'
                    },
                    ticks: { color: '#94a3b8' },
                    grid: { color: '#334155' }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Shares (BUY: +, SELL: -)',
                        color: '#94a3b8'
                    },
                    ticks: { color: '#94a3b8' },
                    grid: { color: '#334155' }
                }
            }
        }
    });
}

// Model Comparison Functions
let availableModelsForComparison = [];
let selectedModelNames = [];

async function loadAvailableModels() {
    try {
        const response = await fetch('/api/trading/models');
        const models = await response.json();
        availableModelsForComparison = models;

        const container = document.getElementById('modelSelectionList');

        if (models.length === 0) {
            container.innerHTML = '<p class="info-text">Henüz eğitilmiş model yok</p>';
            return;
        }

        container.innerHTML = models.map(model => `
            <div class="model-selection-item" onclick="toggleModelSelection('${escapeHtml(model.name)}')" id="select-${escapeHtml(model.name)}">
                <div class="model-selection-name">${escapeHtml(model.name)}</div>
                <div class="model-selection-info">
                    <span><strong>Algoritma:</strong> ${escapeHtml(model.metrics.algorithm || 'N/A')}</span>
                    <span><strong>Return:</strong> ${model.metrics.cumulative_return ? (model.metrics.cumulative_return * 100).toFixed(2) + '%' : 'N/A'}</span>
                    <span><strong>Sharpe:</strong> ${model.metrics.sharpe_ratio?.toFixed(2) || 'N/A'}</span>
                    <span><strong>Trades:</strong> ${model.metrics.total_trades || 0}</span>
                </div>
            </div>
        `).join('');

        // Auto-select all models initially
        selectedModelNames = models.map(m => m.name);
        selectedModelNames.forEach(name => {
            document.getElementById(`select-${name}`)?.classList.add('selected');
        });

    } catch (error) {
        console.error('Model loading error:', error);
        document.getElementById('modelSelectionList').innerHTML =
            '<p class="error">Modeller yüklenemedi</p>';
    }
}

function toggleModelSelection(modelName) {
    const element = document.getElementById(`select-${modelName}`);
    if (!element) return;

    if (selectedModelNames.includes(modelName)) {
        // Deselect
        selectedModelNames = selectedModelNames.filter(n => n !== modelName);
        element.classList.remove('selected');
    } else {
        // Select
        selectedModelNames.push(modelName);
        element.classList.add('selected');
    }
}

function selectAllModels() {
    selectedModelNames = availableModelsForComparison.map(m => m.name);
    availableModelsForComparison.forEach(model => {
        document.getElementById(`select-${model.name}`)?.classList.add('selected');
    });
}

function deselectAllModels() {
    selectedModelNames = [];
    availableModelsForComparison.forEach(model => {
        document.getElementById(`select-${model.name}`)?.classList.remove('selected');
    });
}

async function compareSelectedModels() {
    if (selectedModelNames.length === 0) {
        alert('❌ Lütfen en az bir model seçin!');
        return;
    }

    // Get selected models
    const selectedModels = availableModelsForComparison.filter(m =>
        selectedModelNames.includes(m.name)
    );

    // Show comparison table
    renderComparisonTable(selectedModels);

    // Show comparison chart
    renderComparisonChart(selectedModels);

    // Show the cards
    document.getElementById('comparisonTableCard').style.display = 'block';
    document.getElementById('comparisonChartCard').style.display = 'block';
}

function renderComparisonTable(models) {
    const tbody = document.getElementById('modelsComparisonTable');

    if (models.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8" style="text-align: center;">Model seçilmedi</td></tr>';
        return;
    }

    tbody.innerHTML = models.map(model => {
        const metrics = model.metrics;
        return `
            <tr>
                <td style="font-weight: 600;">${model.name}</td>
                <td><span class="badge">${metrics.algorithm || 'N/A'}</span></td>
                <td style="color: ${(metrics.cumulative_return || 0) >= 0 ? '#10b981' : '#ef4444'}">
                    ${metrics.cumulative_return ? (metrics.cumulative_return * 100).toFixed(2) : 'N/A'}%
                </td>
                <td>${metrics.sharpe_ratio?.toFixed(2) || 'N/A'}</td>
                <td style="color: #ef4444">${metrics.max_drawdown ? (metrics.max_drawdown * 100).toFixed(2) : 'N/A'}%</td>
                <td>${metrics.total_trades || 0}</td>
                <td>$${metrics.final_portfolio_value ? metrics.final_portfolio_value.toFixed(0).replace(/\B(?=(\d{3})+(?!\d))/g, ',') : 'N/A'}</td>
                <td>${metrics.trained_at ? new Date(metrics.trained_at).toLocaleDateString('tr-TR') : 'N/A'}</td>
            </tr>
        `;
    }).join('');
}

function renderComparisonChart(models) {
    const canvas = document.getElementById('algorithmComparisonChart');

    // Destroy existing chart - more thorough cleanup
    if (window.comparisonChartInstance) {
        window.comparisonChartInstance.destroy();
        window.comparisonChartInstance = null;
    }

    // Also check Chart.js internal registry
    const existingChart = Chart.getChart(canvas);
    if (existingChart) {
        existingChart.destroy();
    }

    const ctx = canvas.getContext('2d');
    const labels = models.map(m => m.name.replace('_phase1_', '\n').substring(0, 30));
    const returns = models.map(m => (m.metrics.cumulative_return || 0) * 100);
    const sharpes = models.map(m => m.metrics.sharpe_ratio || 0);
    const trades = models.map(m => m.metrics.total_trades || 0);

    window.comparisonChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Return (%)',
                    data: returns,
                    backgroundColor: 'rgba(16, 185, 129, 0.7)',
                    borderColor: '#10b981',
                    borderWidth: 2,
                    yAxisID: 'y'
                },
                {
                    label: 'Sharpe Ratio',
                    data: sharpes,
                    backgroundColor: 'rgba(59, 130, 246, 0.7)',
                    borderColor: '#3b82f6',
                    borderWidth: 2,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#f8fafc' }
                },
                tooltip: {
                    callbacks: {
                        afterLabel: function(context) {
                            const idx = context.dataIndex;
                            return `Trades: ${trades[idx]}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    ticks: {
                        color: '#94a3b8',
                        font: { size: 10 }
                    },
                    grid: { color: '#334155' }
                },
                y: {
                    type: 'linear',
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Return (%)',
                        color: '#10b981'
                    },
                    ticks: { color: '#10b981' },
                    grid: { color: '#334155' }
                },
                y1: {
                    type: 'linear',
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Sharpe Ratio',
                        color: '#3b82f6'
                    },
                    ticks: { color: '#3b82f6' },
                    grid: { display: false }
                }
            }
        }
    });
}

/**
 * Load Hyperparameter Studies for Selected Algorithm
 */
async function loadHyperparameterStudies() {
    const algorithm = document.getElementById('algorithm').value;
    const hyperparamSelect = document.getElementById('hyperparameter_study');
    const hyperparamSection = document.getElementById('hyperparameter-selection');

    try {
        const response = await fetch(`/api/trading/hyperparameters/${algorithm}`);
        const data = await response.json();

        // Clear existing options except the first one
        hyperparamSelect.innerHTML = '<option value="">Varsayılan parametreleri kullan</option>';

        if (data.studies && data.studies.length > 0) {
            // Add studies as options
            data.studies.forEach(study => {
                const option = document.createElement('option');
                option.value = study.filename;

                // Format the option text
                const date = new Date(study.timestamp);
                const formattedDate = date.toLocaleDateString('tr-TR') + ' ' + date.toLocaleTimeString('tr-TR', {hour: '2-digit', minute: '2-digit'});
                const bestValue = study.best_value ? study.best_value.toFixed(3) : 'N/A';

                option.textContent = `${study.study_name} (Best: ${bestValue}, Trials: ${study.n_trials}) - ${formattedDate}`;
                hyperparamSelect.appendChild(option);
            });

            // Show the section
            hyperparamSection.style.display = 'block';
        } else {
            // No studies found, hide the section
            hyperparamSection.style.display = 'none';
        }
    } catch (error) {
        console.error('Error loading hyperparameter studies:', error);
        // Hide section on error
        hyperparamSection.style.display = 'none';
    }
}

// Run on page load
window.addEventListener('load', init);
