/**
 * Academic Analysis Manager
 * Handles academic analysis UI and interactions
 */

class AcademicAnalysisManager {
    constructor() {
        console.log('AcademicAnalysisManager: Initializing...');
        this.init();
    }

    init() {
        console.log('AcademicAnalysisManager: Setting up event listeners...');
        this.setupEventListeners();
        // Load existing results if available
        console.log('AcademicAnalysisManager: Loading existing results...');
        this.loadExistingResults();
    }

    setupEventListeners() {
        // Generate report button
        const generateBtn = document.getElementById('generate-report-btn');
        if (generateBtn) {
            generateBtn.addEventListener('click', () => this.generateReport());
        }

        // Refresh button
        const refreshBtn = document.getElementById('refresh-analysis-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.loadExistingResults());
        }
    }

    async generateReport() {
        try {
            this.showStatus('🔬 Analiz başlatılıyor...', 'info');

            const response = await fetch('/api/trading/analysis/generate-report', {
                method: 'POST'
            });

            if (!response.ok) {
                throw new Error('Rapor oluşturma başarısız');
            }

            const data = await response.json();

            this.showStatus('✅ Rapor oluşturma başlatıldı! Birkaç dakika sürebilir...', 'success');

            // Poll for results — max 20 retries (#52)
            setTimeout(() => this.checkReportStatus(0), 5000);

        } catch (error) {
            console.error('Error generating report:', error);
            this.showStatus('❌ Hata: ' + error.message, 'error');
        }
    }

    async checkReportStatus(retries = 0) {
        const MAX_RETRIES = 20;  // (#52) prevent infinite polling
        try {
            const response = await fetch('/api/trading/analysis/model-comparison');

            if (response.ok) {
                this.showStatus('✅ Rapor oluşturuldu! Sonuçlar yükleniyor...', 'success');
                await this.loadExistingResults();
                const statusEl = document.getElementById('analysis-status');
                if (statusEl) setTimeout(() => { statusEl.style.display = 'none'; }, 3000);
            } else if (retries < MAX_RETRIES) {
                setTimeout(() => this.checkReportStatus(retries + 1), 5000);
            } else {
                this.showStatus('⏳ Rapor oluşturma zaman aşımına uğradı. Lütfen tekrar deneyin.', 'error');
            }
        } catch (error) {
            console.error('Error checking report status:', error);
            if (retries < MAX_RETRIES) {
                setTimeout(() => this.checkReportStatus(retries + 1), 5000);
            }
        }
    }

    async loadExistingResults() {
        try {
            console.log('AcademicAnalysisManager: Fetching model comparison data...');
            // Load model comparison data
            const comparisonResponse = await fetch('/api/trading/analysis/model-comparison');

            if (!comparisonResponse.ok) {
                console.log('AcademicAnalysisManager: No results - response not ok');
                // No results yet
                this.showEmptyState();
                return;
            }

            const comparisonData = await comparisonResponse.json();
            console.log('AcademicAnalysisManager: Comparison data received:', comparisonData);

            // Check if response has error detail instead of actual data
            if (comparisonData.detail) {
                console.log('AcademicAnalysisManager: No results - detail message received:', comparisonData.detail);
                this.showEmptyState();
                return;
            }

            // Load best models
            console.log('AcademicAnalysisManager: Fetching best models data...');
            const bestModelsResponse = await fetch('/api/trading/analysis/best-models');
            const bestModelsData = await bestModelsResponse.json();
            console.log('AcademicAnalysisManager: Best models data received:', bestModelsData);

            // Hide empty state, show results
            document.getElementById('academic-empty-state').style.display = 'none';
            document.getElementById('best-models-card').style.display = 'block';
            document.getElementById('comparison-table-card').style.display = 'block';
            document.getElementById('charts-grid').style.display = 'block';
            document.getElementById('download-links-card').style.display = 'block';

            // Populate data
            this.displayBestModels(bestModelsData);
            this.displayComparisonTable(comparisonData.models);
            this.renderCharts(comparisonData.models);

        } catch (error) {
            console.error('Error loading results:', error);
            this.showEmptyState();
        }
    }

    displayBestModels(data) {
        // Best Sharpe
        if (data.best_sharpe) {
            document.getElementById('best-sharpe-model').textContent = data.best_sharpe.model;
            document.getElementById('best-sharpe-value').textContent =
                data.best_sharpe.value.toFixed(4);
        }

        // Best Return
        if (data.best_return) {
            document.getElementById('best-return-model').textContent = data.best_return.model;
            document.getElementById('best-return-value').textContent =
                (data.best_return.value * 100).toFixed(2) + '%';
        }

        // Best Drawdown (lowest)
        if (data.lowest_drawdown) {
            document.getElementById('best-drawdown-model').textContent = data.lowest_drawdown.model;
            document.getElementById('best-drawdown-value').textContent =
                (data.lowest_drawdown.value * 100).toFixed(2) + '%';
        }

        // Best Win Rate
        if (data.best_win_rate) {
            document.getElementById('best-winrate-model').textContent = data.best_win_rate.model;
            document.getElementById('best-winrate-value').textContent =
                (data.best_win_rate.value * 100).toFixed(2) + '%';
        }
    }

    displayComparisonTable(models) {
        const tbody = document.getElementById('comparison-table-body');
        if (!tbody) return;

        tbody.innerHTML = '';

        Object.entries(models).forEach(([modelName, metrics]) => {
            const row = document.createElement('tr');

            row.innerHTML = `
                <td style="font-weight: bold;">${AcademicAnalysisManager.escapeHtml(modelName)}</td>
                <td>${(metrics.total_return * 100).toFixed(2)}</td>
                <td>${metrics.sharpe_ratio.toFixed(4)}</td>
                <td>${metrics.sortino_ratio.toFixed(4)}</td>
                <td>${(metrics.max_drawdown * 100).toFixed(2)}</td>
                <td>${(metrics.win_rate * 100).toFixed(2)}</td>
                <td>${metrics.profit_factor.toFixed(2)}</td>
                <td>${metrics.total_trades}</td>
                <td>${metrics.calmar_ratio.toFixed(4)}</td>
            `;

            tbody.appendChild(row);
        });
    }

    renderCharts(models) {
        this.renderPortfolioComparison(models);
        this.renderRiskReturnScatter(models);
        this.renderMetricsBarChart(models);
        this.renderWinRateChart(models);
    }

    renderPortfolioComparison(models) {
        const canvas = document.getElementById('portfolio-comparison-chart');
        if (!canvas) return;

        // Destroy existing chart
        if (canvas.chart) {
            canvas.chart.destroy();
        }

        // (#54) Require at least 1 model with meaningful return data
        const validModels = Object.entries(models).filter(
            ([, metrics]) => metrics.total_return !== undefined && metrics.total_return !== null
        );
        if (validModels.length === 0) {
            canvas.style.display = 'none';
            return;
        }
        canvas.style.display = '';

        const ctx = canvas.getContext('2d');

        // Show normalised portfolio growth: Start=100, End=100*(1+return)
        // Three interpolated points to make chart more readable than bare 2-point line
        const datasets = validModels.map(([name, metrics], index) => {
            const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];
            const endVal = (1 + metrics.total_return) * 100;
            const midVal = (100 + endVal) / 2;
            return {
                label: name,
                data: [100, midVal, endVal],
                borderColor: colors[index % colors.length],
                backgroundColor: 'transparent',
                borderWidth: 2,
                tension: 0.4
            };
        });

        canvas.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: ['Start', 'Mid', 'End'],
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        labels: { color: '#e5e7eb' }
                    }
                },
                scales: {
                    y: {
                        ticks: { color: '#9ca3af' },
                        grid: { color: 'rgba(75, 85, 99, 0.2)' }
                    },
                    x: {
                        ticks: { color: '#9ca3af' },
                        grid: { color: 'rgba(75, 85, 99, 0.2)' }
                    }
                }
            }
        });
    }

    renderRiskReturnScatter(models) {
        const canvas = document.getElementById('risk-return-chart');
        if (!canvas) return;

        if (canvas.chart) {
            canvas.chart.destroy();
        }

        const ctx = canvas.getContext('2d');

        const data = Object.entries(models).map(([name, metrics]) => ({
            x: metrics.annualized_volatility * 100,
            y: metrics.annualized_return * 100,
            label: name
        }));

        canvas.chart = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Models',
                    data: data,
                    backgroundColor: '#3b82f6',
                    borderColor: '#60a5fa',
                    borderWidth: 2,
                    pointRadius: 8,
                    pointHoverRadius: 10
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        labels: { color: '#e5e7eb' }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const point = context.raw;
                                return `${point.label}: Volatility=${point.x.toFixed(2)}%, Return=${point.y.toFixed(2)}%`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Annualized Volatility (%)',
                            color: '#9ca3af'
                        },
                        ticks: { color: '#9ca3af' },
                        grid: { color: 'rgba(75, 85, 99, 0.2)' }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Annualized Return (%)',
                            color: '#9ca3af'
                        },
                        ticks: { color: '#9ca3af' },
                        grid: { color: 'rgba(75, 85, 99, 0.2)' }
                    }
                }
            }
        });
    }

    renderMetricsBarChart(models) {
        const canvas = document.getElementById('metrics-bar-chart');
        if (!canvas) return;

        if (canvas.chart) {
            canvas.chart.destroy();
        }

        const ctx = canvas.getContext('2d');

        const modelNames = Object.keys(models);
        const sharpeRatios = modelNames.map(name => models[name].sharpe_ratio);
        const sortinoRatios = modelNames.map(name => models[name].sortino_ratio);
        const calmarRatios = modelNames.map(name => models[name].calmar_ratio);

        canvas.chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: modelNames,
                datasets: [
                    {
                        label: 'Sharpe Ratio',
                        data: sharpeRatios,
                        backgroundColor: '#3b82f6',
                    },
                    {
                        label: 'Sortino Ratio',
                        data: sortinoRatios,
                        backgroundColor: '#10b981',
                    },
                    {
                        label: 'Calmar Ratio',
                        data: calmarRatios,
                        backgroundColor: '#f59e0b',
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        labels: { color: '#e5e7eb' }
                    }
                },
                scales: {
                    y: {
                        ticks: { color: '#9ca3af' },
                        grid: { color: 'rgba(75, 85, 99, 0.2)' }
                    },
                    x: {
                        ticks: { color: '#9ca3af' },
                        grid: { color: 'rgba(75, 85, 99, 0.2)' }
                    }
                }
            }
        });
    }

    renderWinRateChart(models) {
        const canvas = document.getElementById('winrate-chart');
        if (!canvas) return;

        if (canvas.chart) {
            canvas.chart.destroy();
        }

        const ctx = canvas.getContext('2d');

        const modelNames = Object.keys(models);
        const winRates = modelNames.map(name => models[name].win_rate * 100);
        const profitFactors = modelNames.map(name => models[name].profit_factor);

        canvas.chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: modelNames,
                datasets: [
                    {
                        label: 'Win Rate (%)',
                        data: winRates,
                        backgroundColor: '#10b981',
                        yAxisID: 'y',
                    },
                    {
                        label: 'Profit Factor',
                        data: profitFactors,
                        backgroundColor: '#8b5cf6',
                        yAxisID: 'y1',
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        labels: { color: '#e5e7eb' }
                    }
                },
                scales: {
                    y: {
                        type: 'linear',
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Win Rate (%)',
                            color: '#9ca3af'
                        },
                        ticks: { color: '#9ca3af' },
                        grid: { color: 'rgba(75, 85, 99, 0.2)' }
                    },
                    y1: {
                        type: 'linear',
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Profit Factor',
                            color: '#9ca3af'
                        },
                        ticks: { color: '#9ca3af' },
                        grid: { display: false }
                    },
                    x: {
                        ticks: { color: '#9ca3af' },
                        grid: { color: 'rgba(75, 85, 99, 0.2)' }
                    }
                }
            }
        });
    }

    showStatus(message, type = 'info') {
        const statusDiv = document.getElementById('analysis-status');
        const statusText = document.getElementById('analysis-status-text');

        if (statusDiv && statusText) {
            statusText.textContent = message;
            statusDiv.style.display = 'block';

            // Color based on type
            if (type === 'success') {
                statusDiv.style.borderLeft = '4px solid #10b981';
            } else if (type === 'error') {
                statusDiv.style.borderLeft = '4px solid #ef4444';
            } else {
                statusDiv.style.borderLeft = '4px solid #3b82f6';
            }
        }
    }

    showEmptyState() {
        console.log('AcademicAnalysisManager: Showing empty state...');
        const emptyState = document.getElementById('academic-empty-state');
        if (emptyState) {
            emptyState.style.display = 'block';
            console.log('AcademicAnalysisManager: Empty state display set to block');
        } else {
            console.error('AcademicAnalysisManager: Empty state element not found!');
        }

        // (#53) null-safe style.display access
        ['best-models-card', 'comparison-table-card', 'charts-grid', 'download-links-card'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.style.display = 'none';
        });
        console.log('AcademicAnalysisManager: All result cards hidden');
    }

    static escapeHtml(str) {
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.academicAnalysisManager = new AcademicAnalysisManager();
});
