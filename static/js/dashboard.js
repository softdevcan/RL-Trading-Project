/**
 * RL Trading Dashboard - Main JavaScript
 * Handles all dashboard functionality, API calls, and chart updates
 */

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

    // Training Progress Chart
    const trainCtx = document.getElementById('trainingProgressChart');
    trainingProgressChart = new Chart(trainCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Training Progress (%)',
                data: [],
                borderColor: '#4facfe',
                backgroundColor: 'rgba(79, 172, 254, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            }
        }
    });

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

        try {
            const response = await fetch('/api/trading/train', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });

            const data = await response.json();

            if (response.ok) {
                isTraining = true;
                updateStatus('training');
                document.getElementById('trainButton').disabled = true;
                document.getElementById('progressContainer').style.display = 'block';
                startStatusCheck();
            } else {
                showError(data.detail || 'Eğitim başlatılamadı');
            }
        } catch (error) {
            showError('API bağlantı hatası: ' + error.message);
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
            document.getElementById('progressBar').style.width = progress + '%';
            document.getElementById('progressBar').textContent = progress + '%';
            document.getElementById('stepInfo').textContent =
                `Step: ${data.current_step.toLocaleString()} / ${data.total_steps.toLocaleString()}`;

            // Update training progress chart
            updateTrainingChart(data.current_step, data.progress);
        } else {
            if (isTraining) {
                // Training just finished
                isTraining = false;
                updateStatus('idle');
                document.getElementById('trainButton').disabled = false;
                document.getElementById('progressContainer').style.display = 'none';

                if (data.error) {
                    showError('Eğitim hatası: ' + data.error);
                } else if (data.metrics) {
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
    if (trainingProgressChart.data.labels.length > 50) {
        trainingProgressChart.data.labels.shift();
        trainingProgressChart.data.datasets[0].data.shift();
    }

    trainingProgressChart.data.labels.push(step);
    trainingProgressChart.data.datasets[0].data.push(progress * 100);
    trainingProgressChart.update('none');
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
    // Simulated portfolio growth data
    // In production, this should come from the API with actual portfolio history
    const steps = 100;
    const initialValue = 1000000;
    const finalValue = metrics.final_portfolio_value || initialValue;
    const labels = Array.from({length: steps}, (_, i) => i);

    // Generate realistic portfolio growth curve
    const data = labels.map(i => {
        const progress = i / (steps - 1);
        const noise = (Math.random() - 0.5) * 0.05; // ±5% noise
        return initialValue + (finalValue - initialValue) * progress * (1 + noise);
    });

    performanceChart.data.labels = labels;
    performanceChart.data.datasets[0].data = data;
    performanceChart.update();
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
                <div class="model-name">${model.name}</div>
                <div class="model-meta">
                    Oluşturulma: ${new Date(model.created_at).toLocaleString('tr-TR')}
                </div>
                ${model.metrics.cumulative_return ? `
                    <div class="model-meta">
                        Return: ${(model.metrics.cumulative_return * 100).toFixed(2)}% |
                        Sharpe: ${model.metrics.sharpe_ratio?.toFixed(4) || 'N/A'}
                    </div>
                ` : ''}
            </div>
        `).join('');
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
            <div style="font-weight: 600; margin-bottom: 10px;">${selectedModel.name}</div>
            <div style="font-size: 14px; color: #666;">
                Algoritma: ${selectedModel.metrics.algorithm || 'N/A'}<br>
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
                <td>${model.name}</td>
                <td>${model.metrics.algorithm || 'N/A'}</td>
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
function showError(message) {
    const errorContainer = document.getElementById('errorContainer');
    errorContainer.innerHTML = `<div class="error">${message}</div>`;
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
    console.log('Dashboard initialized successfully!');
}

// Run on page load
window.addEventListener('load', init);
