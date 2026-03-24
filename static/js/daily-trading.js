/**
 * Daily Trading Manager
 * Handles UI interactions and API calls for daily trading decisions
 */

class DailyTradingManager {
    constructor() {
        console.log('DailyTradingManager: Initializing...');
        this.currentDecision = null;
        this.portfolioChart = null;
        this.init();
    }

    init() {
        console.log('DailyTradingManager: Running init...');
        this.loadModels();
        this.setupEventListeners();
        this.loadLatestPortfolio();
        this.setDefaultDate();
    }

    setDefaultDate() {
        // Set trading date to last weekday (more likely to have data)
        const date = new Date();

        // Go back until we find a weekday (Mon-Fri)
        while (date.getDay() === 0 || date.getDay() === 6) {
            date.setDate(date.getDate() - 1);
        }

        // Go back 1-2 more days to ensure market data is available
        date.setDate(date.getDate() - 2);

        // Skip weekend again if needed
        while (date.getDay() === 0 || date.getDay() === 6) {
            date.setDate(date.getDate() - 1);
        }

        const dateStr = date.toISOString().split('T')[0];

        const dateInput = document.getElementById('trading-date');
        if (dateInput) {
            dateInput.value = dateStr;
            console.log('Default trading date set to:', dateStr);
        }
    }

    setupEventListeners() {
        console.log('DailyTradingManager: Setting up event listeners...');

        // Slider update - FIX: Use correct ID 'max-shares-value' instead of 'max-shares-display'
        const slider = document.getElementById('max-shares-slider');
        const display = document.getElementById('max-shares-value');
        if (slider && display) {
            slider.addEventListener('input', (e) => {
                display.textContent = e.target.value;
                console.log('Slider value updated:', e.target.value);
            });
            console.log('Slider event listener attached');
        } else {
            console.error('Slider or display element not found!', {slider, display});
        }

        // Get Decision button - FIX: Use correct ID 'get-daily-decision'
        const getDecisionBtn = document.getElementById('get-daily-decision');
        if (getDecisionBtn) {
            getDecisionBtn.addEventListener('click', () => this.getDecision());
            console.log('Get decision button listener attached');
        } else {
            console.error('Get decision button not found!');
        }

        // Load Portfolio button - FIX: Use correct ID 'load-last-portfolio'
        const loadPortfolioBtn = document.getElementById('load-last-portfolio');
        if (loadPortfolioBtn) {
            loadPortfolioBtn.addEventListener('click', () => this.loadLatestPortfolio());
            console.log('Load portfolio button listener attached');
        } else {
            console.error('Load portfolio button not found!');
        }

        // Apply Decision button
        const applyDecisionBtn = document.getElementById('apply-decisions');
        if (applyDecisionBtn) {
            applyDecisionBtn.addEventListener('click', () => this.applyDecision());
            console.log('Apply decisions button listener attached');
        }

        // Export button
        const exportBtn = document.getElementById('export-decisions');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => this.exportDecision());
            console.log('Export button listener attached');
        }
    }

    async loadModels() {
        try {
            console.log('DailyTradingManager: Loading models...');
            const response = await fetch('/api/trading/models');
            const data = await response.json();
            console.log('Models data received:', data);

            // FIX: API returns array directly, not {models: [...]}
            const select = document.getElementById('live-model-select');
            if (select && Array.isArray(data) && data.length > 0) {
                select.innerHTML = '<option value="">-- Model Seç --</option>';
                data.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model.name;
                    option.textContent = model.name;
                    select.appendChild(option);
                });
                console.log(`✅ Loaded ${data.length} models into select`);
            } else {
                console.error('Model select element not found or no models in data!', {select, data});
            }
        } catch (error) {
            console.error('Error loading models:', error);
            this.showError('Modeller yüklenirken hata oluştu');
        }
    }

    async loadLatestPortfolio() {
        try {
            console.log('Loading latest portfolio...');
            const response = await fetch('/api/trading/latest-portfolio');
            const data = await response.json();
            console.log('Portfolio data received:', data);

            if (data.portfolio) {
                // Update balance - FIX: Use correct ID 'current-balance'
                const balanceInput = document.getElementById('current-balance');
                if (balanceInput) {
                    balanceInput.value = data.portfolio.balance;
                    console.log('Balance updated:', data.portfolio.balance);
                } else {
                    console.error('Balance input element not found!');
                }

                // Update shares
                Object.entries(data.portfolio.shares).forEach(([symbol, shares]) => {
                    const input = document.getElementById(`shares-${symbol.replace('.IS', '')}`);
                    if (input) {
                        input.value = shares;
                        console.log(`${symbol} shares updated:`, shares);
                    } else {
                        console.error(`Shares input not found for ${symbol}`);
                    }
                });

                this.showSuccess('Portfolio yüklendi');
            }
        } catch (error) {
            console.error('Error loading portfolio:', error);
            this.showError('Portfolio yüklenirken hata oluştu');
        }
    }

    async getDecision() {
        try {
            // Validate inputs - FIX: Use correct ID 'live-model-select'
            const modelName = document.getElementById('live-model-select')?.value;
            console.log('Selected model:', modelName);
            if (!modelName) {
                this.showError('Lütfen bir model seçin');
                return;
            }

            // Get risk mode
            const riskMode = document.querySelector('input[name="risk-mode"]:checked')?.value || 'moderate';
            console.log('Risk mode:', riskMode);

            // Get max shares
            const maxShares = parseInt(document.getElementById('max-shares-slider')?.value || 100);
            console.log('Max shares:', maxShares);

            // Get trading date
            const tradingDate = document.getElementById('trading-date')?.value;
            if (!tradingDate) {
                this.showError('Lütfen bir işlem tarihi seçin');
                return;
            }
            console.log('Trading date:', tradingDate);

            // Get balance - FIX: Use correct ID 'current-balance'
            const balance = parseFloat(document.getElementById('current-balance')?.value || 1000000);
            console.log('Balance:', balance);

            // Get shares - PHASE1_SYMBOLS (must match training data)
            const symbols = ['AKBNK', 'THYAO', 'TUPRS', 'BIMAS', 'ASELS'];
            const shares = {};
            symbols.forEach(symbol => {
                const input = document.getElementById(`shares-${symbol}`);
                shares[`${symbol}.IS`] = parseInt(input?.value || 0);
            });

            // Show loading state
            this.showLoading();

            // Make API call
            const response = await fetch('/api/trading/daily-decision', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model_name: modelName,
                    balance: balance,
                    shares: shares,
                    risk_mode: riskMode,
                    max_shares_per_trade: maxShares,
                    date: tradingDate
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'API error');
            }

            const data = await response.json();
            this.currentDecision = data;
            this.displayDecision(data);

        } catch (error) {
            console.error('Error getting decision:', error);
            this.showError('Karar alınırken hata oluştu: ' + error.message);
            this.hideLoading();
        }
    }

    displayDecision(data) {
        this.hideLoading();

        // Show decisions container
        const emptyState = document.getElementById('empty-state');
        const decisionsContainer = document.getElementById('decisions-container');
        if (emptyState) emptyState.style.display = 'none';
        if (decisionsContainer) decisionsContainer.style.display = 'block';

        // Update header
        const dateElement = document.getElementById('decision-date');
        const portfolioBeforeElement = document.getElementById('portfolio-before');
        const portfolioAfterElement = document.getElementById('portfolio-after');
        const returnElement = document.getElementById('daily-return-badge');

        // Show date (with warning if actual_date differs from requested)
        if (dateElement) {
            let dateText = data.date;
            if (data.summary?.actual_date && data.summary.actual_date !== data.summary.requested_date) {
                dateText += ` (⚠️ ${data.summary.requested_date} için veri yok, ${data.summary.actual_date} kullanıldı)`;
            }
            dateElement.textContent = dateText;
        }
        if (portfolioBeforeElement) {
            portfolioBeforeElement.textContent =
                data.portfolio_before.portfolio_value.toLocaleString('tr-TR', {
                    style: 'currency',
                    currency: 'TRY',
                    minimumFractionDigits: 2
                });
        }
        if (portfolioAfterElement) {
            portfolioAfterElement.textContent =
                data.portfolio_after.portfolio_value.toLocaleString('tr-TR', {
                    style: 'currency',
                    currency: 'TRY',
                    minimumFractionDigits: 2
                });
        }
        if (returnElement) {
            const returnPct = data.summary.daily_return_pct || 0;
            returnElement.textContent = returnPct.toFixed(2) + '%';
            returnElement.className = 'return-badge ' + (returnPct >= 0 ? 'positive' : 'negative');
        }

        // Update summary stats (with null checks)
        const totalTradesEl = document.getElementById('summary-total-trades');
        const commissionEl = document.getElementById('summary-total-commission');

        if (totalTradesEl) {
            totalTradesEl.textContent = data.summary.total_trades || 0;
        }
        if (commissionEl) {
            commissionEl.textContent = (data.summary.total_commission || 0).toFixed(2) + ' ₺';
        }

        // Optional stats (may not exist in HTML or backend)
        const buysEl = document.getElementById('summary-buys');
        const sellsEl = document.getElementById('summary-sells');
        const holdsEl = document.getElementById('summary-holds');

        if (buysEl && data.summary.buys !== undefined) {
            buysEl.textContent = data.summary.buys;
        }
        if (sellsEl && data.summary.sells !== undefined) {
            sellsEl.textContent = data.summary.sells;
        }
        if (holdsEl && data.summary.holds !== undefined) {
            holdsEl.textContent = data.summary.holds;
        }

        // Update table
        this.updateDecisionsTable(data.decisions);

        // Enable apply button if there are executed trades
        const applyBtn = document.getElementById('apply-decision-btn');
        if (applyBtn) {
            const hasExecutedTrades = data.decisions.some(d => d.executed);
            applyBtn.disabled = !hasExecutedTrades;
        }
    }

    updateDecisionsTable(decisions) {
        const tbody = document.getElementById('decisions-table-body');
        if (!tbody) return;

        tbody.innerHTML = '';

        decisions.forEach(decision => {
            const row = document.createElement('tr');

            // Symbol
            const symbolCell = document.createElement('td');
            symbolCell.textContent = decision.symbol;
            row.appendChild(symbolCell);

            // Action badge
            const actionCell = document.createElement('td');
            const badge = document.createElement('span');
            badge.className = `action-badge action-${decision.action.toLowerCase()}`;
            badge.textContent = decision.action;
            actionCell.appendChild(badge);
            row.appendChild(actionCell);

            // Signal with bar
            const signalCell = document.createElement('td');
            const signalContainer = document.createElement('div');
            signalContainer.className = 'signal-container';

            const signalBar = document.createElement('div');
            signalBar.className = 'signal-bar';
            const signalValue = Math.abs(decision.raw_signal);
            signalBar.style.width = (signalValue * 100) + '%';

            if (decision.action === 'BUY') {
                signalBar.style.backgroundColor = '#10b981';
            } else if (decision.action === 'SELL') {
                signalBar.style.backgroundColor = '#ef4444';
            } else {
                signalBar.style.backgroundColor = '#6b7280';
            }

            const signalText = document.createElement('span');
            signalText.className = 'signal-text';
            signalText.textContent = decision.raw_signal.toFixed(3);

            signalContainer.appendChild(signalBar);
            signalContainer.appendChild(signalText);
            signalCell.appendChild(signalContainer);
            row.appendChild(signalCell);

            // Shares
            const sharesCell = document.createElement('td');
            sharesCell.textContent = decision.shares;
            row.appendChild(sharesCell);

            // Price
            const priceCell = document.createElement('td');
            priceCell.textContent = decision.price.toFixed(2) + ' ₺';
            row.appendChild(priceCell);

            // Amount
            const amountCell = document.createElement('td');
            const amount = decision.cost || decision.revenue || 0;
            amountCell.textContent = amount.toFixed(2) + ' ₺';
            if (decision.action === 'BUY') {
                amountCell.style.color = '#ef4444';
            } else if (decision.action === 'SELL') {
                amountCell.style.color = '#10b981';
            }
            row.appendChild(amountCell);

            // Commission
            const commissionCell = document.createElement('td');
            commissionCell.textContent = decision.commission.toFixed(2) + ' ₺';
            row.appendChild(commissionCell);

            // Reason
            const reasonCell = document.createElement('td');
            reasonCell.textContent = decision.reason;
            row.appendChild(reasonCell);

            // Executed
            const executedCell = document.createElement('td');
            executedCell.textContent = decision.executed ? '✅' : '❌';
            row.appendChild(executedCell);

            tbody.appendChild(row);
        });
    }

    async applyDecision() {
        if (!this.currentDecision) {
            this.showError('Önce bir karar almalısınız');
            return;
        }

        try {
            const response = await fetch('/api/trading/apply-decision', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    date: this.currentDecision.date
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'API error');
            }

            const data = await response.json();
            this.showSuccess('Karar portfolio geçmişine eklendi');

            // Reload portfolio history chart
            await this.loadPortfolioHistory();

        } catch (error) {
            console.error('Error applying decision:', error);
            this.showError('Karar uygulanırken hata oluştu: ' + error.message);
        }
    }

    async loadPortfolioHistory() {
        try {
            const response = await fetch('/api/trading/portfolio-history?days=30');
            const data = await response.json();

            if (data.dates && data.dates.length > 0) {
                this.renderPortfolioChart(data);
            }
        } catch (error) {
            console.error('Error loading portfolio history:', error);
        }
    }

    renderPortfolioChart(data) {
        const canvas = document.getElementById('portfolio-chart');
        if (!canvas) return;

        // Destroy existing chart
        if (this.portfolioChart) {
            this.portfolioChart.destroy();
        }

        const ctx = canvas.getContext('2d');
        this.portfolioChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.dates,
                datasets: [{
                    label: 'Portfolio Değeri',
                    data: data.portfolio_values,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        labels: {
                            color: '#e5e7eb'
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        callbacks: {
                            label: function(context) {
                                return 'Portfolio: ' + context.parsed.y.toLocaleString('tr-TR', {
                                    style: 'currency',
                                    currency: 'TRY'
                                });
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            color: '#9ca3af'
                        },
                        grid: {
                            color: 'rgba(75, 85, 99, 0.2)'
                        }
                    },
                    y: {
                        ticks: {
                            color: '#9ca3af',
                            callback: function(value) {
                                return value.toLocaleString('tr-TR');
                            }
                        },
                        grid: {
                            color: 'rgba(75, 85, 99, 0.2)'
                        }
                    }
                }
            }
        });
    }

    exportDecision() {
        if (!this.currentDecision) {
            this.showError('Önce bir karar almalısınız');
            return;
        }

        const dataStr = JSON.stringify(this.currentDecision, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);

        const link = document.createElement('a');
        link.href = url;
        link.download = `trading_decision_${this.currentDecision.date}.json`;
        link.click();

        URL.revokeObjectURL(url);
        this.showSuccess('Karar JSON olarak indirildi');
    }

    showLoading() {
        const loadingState = document.getElementById('loading-state');
        const emptyState = document.getElementById('empty-state');
        const decisionsContainer = document.getElementById('decisions-container');

        if (loadingState) loadingState.style.display = 'flex';
        if (emptyState) emptyState.style.display = 'none';
        if (decisionsContainer) decisionsContainer.style.display = 'none';
    }

    hideLoading() {
        const loadingState = document.getElementById('loading-state');
        if (loadingState) loadingState.style.display = 'none';
    }

    showError(message) {
        // Simple alert for now - could be replaced with a toast notification
        alert('❌ ' + message);
    }

    showSuccess(message) {
        // Simple alert for now - could be replaced with a toast notification
        alert('✅ ' + message);
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.dailyTradingManager = new DailyTradingManager();
});
