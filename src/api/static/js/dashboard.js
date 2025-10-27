// Dashboard JavaScript - Main functionality
// Author: ramkumarjayakumar
// Date: 2025-10-27

const API_BASE = 'http://127.0.0.1:8000/api/v1';
let monitoringInterval = null;

// ============ INITIALIZATION ============
document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard loaded');
    initializeSystemStatus();
    loadPage('overview');
    checkSystemHealth();
    setupEventListeners();
    loadDashboardOverview();
    loadSystemLogs();
});

function initializeSystemStatus() {
    const statusDot = document.getElementById('systemStatus');
    const statusText = document.getElementById('statusText');
    if (statusDot && statusText) {
        statusDot.style.backgroundColor = '#10b981';
        statusText.textContent = 'System Healthy';
    }
}

function setupEventListeners() {
    const menuToggle = document.getElementById('menuToggle');
    const refreshBtn = document.getElementById('refreshBtn');
    if (menuToggle) menuToggle.addEventListener('click', toggleSidebar);
    if (refreshBtn) refreshBtn.addEventListener('click', refreshData);
}

function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    if (sidebar) sidebar.classList.toggle('active');
}

// ============ PAGE NAVIGATION ============
function loadPage(pageName) {
    console.log('Loading page:', pageName);

    // Hide all pages
    const pages = document.querySelectorAll('.page');
    pages.forEach(function(page) {
        page.classList.remove('active');
    });

    // Show selected page
    const pageId = pageName + '-page';
    const page = document.getElementById(pageId);
    if (page) {
        page.classList.add('active');
        console.log('Page activated:', pageId);
    }

    // Update nav links
    const links = document.querySelectorAll('.nav-link');
    links.forEach(function(link) {
        link.classList.remove('active');
        if (link.getAttribute('href') === '#' + pageName) {
            link.classList.add('active');
        }
    });

    // Update page title
    const titles = {
        'overview': 'Dashboard Overview',
        'data': 'Data Pipeline',
        'ml': 'ML Pipeline',
        'monitor': 'Real-time Monitoring',
        'control': 'Control Panel',
        'logs': 'System Logs'
    };

    const pageTitle = document.getElementById('pageTitle');
    if (pageTitle) {
        pageTitle.textContent = titles[pageName] || 'Dashboard';
    }

    // Close sidebar on mobile
    if (window.innerWidth <= 768) {
        const sidebar = document.querySelector('.sidebar');
        if (sidebar) sidebar.classList.remove('active');
    }

    return false;
}

// ============ SYSTEM HEALTH ============
async function checkSystemHealth() {
    try {
        const response = await fetch(API_BASE + '/health');
        const data = await response.json();

        const statusDot = document.getElementById('systemStatus');
        const statusText = document.getElementById('statusText');

        if (data.status === 'success' || response.ok) {
            if (statusDot) statusDot.style.backgroundColor = '#10b981';
            if (statusText) statusText.textContent = 'System Healthy';
        } else {
            if (statusDot) statusDot.style.backgroundColor = '#ef4444';
            if (statusText) statusText.textContent = 'System Error';
        }
    } catch (error) {
        console.error('Error checking system health:', error);
        const statusDot = document.getElementById('systemStatus');
        const statusText = document.getElementById('statusText');
        if (statusDot) statusDot.style.backgroundColor = '#ef4444';
        if (statusText) statusText.textContent = 'Connection Error';
    }

    // Check again every 30 seconds
    setTimeout(checkSystemHealth, 30000);
}

// ============ DASHBOARD OVERVIEW ============
async function loadDashboardOverview() {
    showSpinner();

    try {
        const response = await fetch(API_BASE + '/dashboard/overview');
        const result = await response.json();

        if (result.status === 'success') {
            const data = result.data;

            // Update metrics
            const healthStatus = document.getElementById('healthStatus');
            if (healthStatus) healthStatus.textContent = data.system_health.status;

            const recordsProcessed = document.getElementById('recordsProcessed');
            if (recordsProcessed) recordsProcessed.textContent = data.data_pipeline.total_records_processed.toLocaleString();

            const featuresCount = document.getElementById('featuresCount');
            if (featuresCount) featuresCount.textContent = data.data_pipeline.features_engineered;

            if (data.ml_pipeline.best_model) {
                const bestModel = document.getElementById('bestModel');
                if (bestModel) bestModel.textContent = data.ml_pipeline.best_model;

                const metrics = data.ml_pipeline.model_metrics[data.ml_pipeline.best_model];
                if (metrics) {
                    const bestAccuracy = document.getElementById('bestAccuracy');
                    if (bestAccuracy) bestAccuracy.textContent = metrics.accuracy.toFixed(2) + '%';
                }
            }

            // Update model comparison
            updateModelComparison(data.ml_pipeline.model_metrics);

            // Update system resources
            await loadSystemStats();

            showToast('Dashboard updated successfully', 'success');
        }
    } catch (error) {
        console.error('Error loading dashboard overview:', error);
        showToast('Error: ' + error.message, 'error');
    }

    hideSpinner();
}

function updateModelComparison(metrics) {
    const container = document.getElementById('modelComparison');
    if (!container) return;

    container.innerHTML = '';

    for (const modelName in metrics) {
        const modelMetrics = metrics[modelName];
        const modelCard = document.createElement('div');
        modelCard.className = 'model-card';
        modelCard.innerHTML = '<h4>' + modelName + '</h4>' +
            '<div class="model-metric"><span class="model-metric-label">Accuracy</span><span class="model-metric-value">' + modelMetrics.accuracy.toFixed(2) + '%</span></div>' +
            '<div class="model-metric"><span class="model-metric-label">Precision</span><span class="model-metric-value">' + modelMetrics.precision.toFixed(2) + '%</span></div>' +
            '<div class="model-metric"><span class="model-metric-label">Recall</span><span class="model-metric-value">' + modelMetrics.recall.toFixed(2) + '%</span></div>' +
            '<div class="model-metric"><span class="model-metric-label">F1 Score</span><span class="model-metric-value">' + modelMetrics.f1_score.toFixed(2) + '%</span></div>';
        container.appendChild(modelCard);
    }
}

async function loadSystemStats() {
    try {
        const response = await fetch(API_BASE + '/dashboard/system-stats');
        const result = await response.json();

        if (result.status === 'success') {
            const stats = result.data;

            // Update progress bars
            updateProgressBar('cpuProgress', stats.cpu.usage_percent);
            updateProgressBar('memoryProgress', stats.memory.percent);
            updateProgressBar('diskProgress', stats.disk.percent);

            // Update values
            const cpuValue = document.getElementById('cpuValue');
            if (cpuValue) cpuValue.textContent = stats.cpu.usage_percent.toFixed(1) + '%';

            const memoryValue = document.getElementById('memoryValue');
            if (memoryValue) memoryValue.textContent = stats.memory.percent.toFixed(1) + '%';

            const diskValue = document.getElementById('diskValue');
            if (diskValue) diskValue.textContent = stats.disk.percent.toFixed(1) + '%';
        }
    } catch (error) {
        console.error('Error loading system stats:', error);
    }
}

function updateProgressBar(elementId, value) {
    const element = document.getElementById(elementId);
    if (element) {
        element.style.width = Math.min(value, 100) + '%';
    }
}

// ============ DATA PIPELINE ============
async function loadCorrelationData() {
    showSpinner();
    try {
        const response = await fetch(API_BASE + '/data/correlation-analysis');
        const result = await response.json();

        const container = document.getElementById('correlationData');
        if (container) {
            if (result.status === 'success') {
                container.innerHTML = '<pre>' + JSON.stringify(result.data, null, 2) + '</pre>';
                showToast('Correlation data loaded', 'success');
            } else {
                container.innerHTML = '<p style="color: red;">Error loading data</p>';
            }
        }
    } catch (error) {
        console.error('Error loading correlation data:', error);
        showToast('Error: ' + error.message, 'error');
    }
    hideSpinner();
}

async function loadStatisticalData() {
    showSpinner();
    try {
        const response = await fetch(API_BASE + '/data/statistical-analysis');
        const result = await response.json();

        const container = document.getElementById('statisticalData');
        if (container) {
            if (result.status === 'success') {
                container.innerHTML = '<pre>' + JSON.stringify(result.data, null, 2) + '</pre>';
                showToast('Statistical data loaded', 'success');
            } else {
                container.innerHTML = '<p style="color: red;">Error loading data</p>';
            }
        }
    } catch (error) {
        console.error('Error loading statistical data:', error);
        showToast('Error: ' + error.message, 'error');
    }
    hideSpinner();
}

async function loadDriftData() {
    showSpinner();
    try {
        const response = await fetch(API_BASE + '/data/drift-analysis');
        const result = await response.json();

        const container = document.getElementById('driftData');
        if (container) {
            if (result.status === 'success') {
                container.innerHTML = '<p><strong>Latest Drift File:</strong> ' + result.data.latest_file + '</p><pre>' + JSON.stringify(result.data.data, null, 2) + '</pre>';
                showToast('Drift data loaded', 'success');
            } else {
                container.innerHTML = '<p style="color: red;">Error loading data</p>';
            }
        }
    } catch (error) {
        console.error('Error loading drift data:', error);
        showToast('Error: ' + error.message, 'error');
    }
    hideSpinner();
}

async function loadDriftHistory() {
    showSpinner();
    try {
        const response = await fetch(API_BASE + '/data/drift-history');
        const result = await response.json();

        const container = document.getElementById('driftData');
        if (container) {
            if (result.status === 'success') {
                let html = '<p><strong>Total Drift Records:</strong> ' + result.data.count + '</p>';
                result.data.data.forEach(function(item) {
                    html += '<h4>' + item.timestamp + '</h4><pre>' + JSON.stringify(item.data, null, 2) + '</pre>';
                });
                container.innerHTML = html;
                showToast('Drift history loaded', 'success');
            } else {
                container.innerHTML = '<p style="color: red;">Error loading data</p>';
            }
        }
    } catch (error) {
        console.error('Error loading drift history:', error);
        showToast('Error: ' + error.message, 'error');
    }
    hideSpinner();
}

async function loadValidationReport() {
    showSpinner();
    try {
        const response = await fetch(API_BASE + '/data/validation-report');
        const result = await response.json();

        const container = document.getElementById('validationReport');
        if (container) {
            if (result.status === 'success') {
                container.innerHTML = '<pre>' + result.data.report + '</pre>';
                showToast('Validation report loaded', 'success');
            } else {
                container.innerHTML = '<p style="color: red;">Error loading data</p>';
            }
        }
    } catch (error) {
        console.error('Error loading validation report:', error);
        showToast('Error: ' + error.message, 'error');
    }
    hideSpinner();
}

// ============ ML PIPELINE ============
async function loadEvaluationResults() {
    showSpinner();
    try {
        const response = await fetch(API_BASE + '/ml/evaluation-results');
        const result = await response.json();

        const container = document.getElementById('evaluationResults');
        if (container) {
            if (result.status === 'success') {
                container.innerHTML = '<pre>' + JSON.stringify(result.data, null, 2) + '</pre>';
                showToast('Evaluation results loaded', 'success');
            } else {
                container.innerHTML = '<p style="color: red;">Error loading data</p>';
            }
        }
    } catch (error) {
        console.error('Error loading evaluation results:', error);
        showToast('Error: ' + error.message, 'error');
    }
    hideSpinner();
}

async function loadFeatureImportance() {
    showSpinner();
    try {
        const response = await fetch(API_BASE + '/ml/feature-importance');
        const result = await response.json();

        const container = document.getElementById('featureImportance');
        if (container) {
            if (result.status === 'success') {
                container.innerHTML = '<pre>' + JSON.stringify(result.data, null, 2) + '</pre>';
                showToast('Feature importance loaded', 'success');
            } else {
                container.innerHTML = '<p style="color: red;">Error loading data</p>';
            }
        }
    } catch (error) {
        console.error('Error loading feature importance:', error);
        showToast('Error: ' + error.message, 'error');
    }
    hideSpinner();
}

async function loadModelHealth() {
    showSpinner();
    try {
        const response = await fetch(API_BASE + '/ml/model-health');
        const result = await response.json();

        const container = document.getElementById('modelHealth');
        if (container) {
            if (result.status === 'success') {
                let html = '<p><strong>Last Updated:</strong> ' + result.data.last_updated + '</p>';
                result.data.models.forEach(function(model) {
                    html += '<div style="margin: 10px 0; padding: 10px; background: #f0f0f0; border-radius: 5px;">' +
                        '<h4>' + model.name + '</h4>' +
                        '<p><strong>Status:</strong> ' + model.health_status + '</p>' +
                        '<p><strong>Health Score:</strong> ' + model.health_score + '/100</p>' +
                        '<p><strong>Accuracy:</strong> ' + (model.accuracy * 100).toFixed(2) + '%</p>' +
                        '</div>';
                });
                container.innerHTML = html;
                showToast('Model health loaded', 'success');
            } else {
                container.innerHTML = '<p style="color: red;">Error loading data</p>';
            }
        }
    } catch (error) {
        console.error('Error loading model health:', error);
        showToast('Error: ' + error.message, 'error');
    }
    hideSpinner();
}

async function loadMLOpsMetrics() {
    showSpinner();
    try {
        const response = await fetch(API_BASE + '/ml/mlops-metrics');
        const result = await response.json();

        const container = document.getElementById('mlopsMetrics');
        if (container) {
            if (result.status === 'success') {
                container.innerHTML = '<pre>' + JSON.stringify(result.data, null, 2) + '</pre>';
                showToast('MLOps metrics loaded', 'success');
            } else {
                container.innerHTML = '<p style="color: red;">Error loading data</p>';
            }
        }
    } catch (error) {
        console.error('Error loading MLOps metrics:', error);
        showToast('Error: ' + error.message, 'error');
    }
    hideSpinner();
}

async function loadXGBoostMetrics() {
    showSpinner();
    try {
        const response = await fetch(API_BASE + '/ml/mlops-metrics?model=xgboost');
        const result = await response.json();

        const container = document.getElementById('mlopsMetrics');
        if (container) {
            if (result.status === 'success') {
                container.innerHTML = '<pre>' + JSON.stringify(result.data, null, 2) + '</pre>';
                showToast('XGBoost metrics loaded', 'success');
            } else {
                container.innerHTML = '<p style="color: red;">Error loading data</p>';
            }
        }
    } catch (error) {
        console.error('Error loading XGBoost metrics:', error);
        showToast('Error: ' + error.message, 'error');
    }
    hideSpinner();
}

// ============ PIPELINE CONTROL ============
async function triggerDataPipeline() {
    showToast('Data pipeline trigger not yet implemented in backend', 'info');
}

async function triggerMLPipeline() {
    showToast('ML pipeline trigger not yet implemented in backend', 'info');
}

async function triggerDataPipelineExtended() {
    showSpinner();
    try {
        const response = await fetch(API_BASE + '/dashboard/trigger/data-pipeline', { method: 'POST' });
        const result = await response.json();

        const statusDiv = document.getElementById('dataPipelineStatus');
        if (statusDiv) {
            if (result.status === 'success') {
                statusDiv.innerHTML = '<p style="color: green;"><strong>✓ Success</strong></p>' +
                    '<p>Process ID: ' + result.data.process_id + '</p>' +
                    '<p>Message: ' + result.data.message + '</p>';
                showToast('Data pipeline triggered successfully', 'success');
            } else {
                statusDiv.innerHTML = '<p style="color: red;"><strong>✗ Error:</strong> ' + result.data.message + '</p>';
                showToast('Error: ' + result.data.message, 'error');
            }
        }
    } catch (error) {
        console.error('Error triggering data pipeline:', error);
        showToast('Error: ' + error.message, 'error');
    }
    hideSpinner();
}

async function triggerMLPipelineExtended() {
    showSpinner();
    try {
        const response = await fetch(API_BASE + '/dashboard/trigger/ml-pipeline', { method: 'POST' });
        const result = await response.json();

        const statusDiv = document.getElementById('mlPipelineStatus');
        if (statusDiv) {
            if (result.status === 'success') {
                statusDiv.innerHTML = '<p style="color: green;"><strong>✓ Success</strong></p>' +
                    '<p>Process ID: ' + result.data.process_id + '</p>' +
                    '<p>Message: ' + result.data.message + '</p>';
                showToast('ML pipeline triggered successfully', 'success');
            } else {
                statusDiv.innerHTML = '<p style="color: red;"><strong>✗ Error:</strong> ' + result.data.message + '</p>';
                showToast('Error: ' + result.data.message, 'error');
            }
        }
    } catch (error) {
        console.error('Error triggering ML pipeline:', error);
        showToast('Error: ' + error.message, 'error');
    }
    hideSpinner();
}

async function loadPipelineLogs(limit) {
    showSpinner();
    limit = limit || 10;
    try {
        const response = await fetch(API_BASE + '/ml/pipeline-logs?limit=' + limit);
        const result = await response.json();

        const container = document.getElementById('pipelineLogs');
        if (container) {
            if (result.status === 'success') {
                let html = '<p><strong>Total Logs:</strong> ' + result.data.count + '</p>';
                result.data.data.forEach(function(log) {
                    html += '<p><strong>Timestamp:</strong> ' + log.timestamp + '</p>' +
                        '<p><strong>File:</strong> ' + log.file + '</p>' +
                        '<p><strong>Size:</strong> ' + log.size_kb.toFixed(2) + ' KB</p>' +
                        '<hr>';
                });
                container.innerHTML = html;
                showToast('Pipeline logs loaded', 'success');
            } else {
                container.innerHTML = '<p style="color: red;">Error loading logs</p>';
            }
        }
    } catch (error) {
        console.error('Error loading pipeline logs:', error);
        showToast('Error: ' + error.message, 'error');
    }
    hideSpinner();
}

// ============ MONITORING ============
function startMonitoring() {
    if (monitoringInterval) return;

    showToast('Monitoring started', 'info');
    monitoringInterval = setInterval(updateMonitoringData, 5000);
    updateMonitoringData();
}

function stopMonitoring() {
    if (monitoringInterval) {
        clearInterval(monitoringInterval);
        monitoringInterval = null;
        showToast('Monitoring stopped', 'info');
    }
}

function toggleAutoRefresh() {
    const checkbox = document.getElementById('autoRefresh');
    if (checkbox && checkbox.checked) {
        startMonitoring();
    } else {
        stopMonitoring();
    }
}

async function updateMonitoringData() {
    try {
        const response = await fetch(API_BASE + '/dashboard/system-stats');
        const result = await response.json();

        if (result.status === 'success') {
            const stats = result.data;

            const monCPU = document.getElementById('monCPU');
            if (monCPU) monCPU.textContent = stats.cpu.usage_percent.toFixed(1) + '%';

            const monMemory = document.getElementById('monMemory');
            if (monMemory) monMemory.textContent = stats.memory.percent.toFixed(1) + '%';

            const monDisk = document.getElementById('monDisk');
            if (monDisk) monDisk.textContent = stats.disk.percent.toFixed(1) + '%';
        }
    } catch (error) {
        console.error('Error updating monitoring data:', error);
    }
}

// ============ LOGS ============
async function loadSystemLogs() {
    try {
        const response = await fetch(API_BASE + '/ml/pipeline-logs?limit=20');
        const result = await response.json();

        const logOutput = document.getElementById('logOutput');
        if (logOutput) {
            logOutput.innerHTML = '';

            if (result.status === 'success' && result.data.data && result.data.data.length > 0) {
                result.data.data.forEach(function(log) {
                    const logLine = document.createElement('div');
                    logLine.className = 'log-line';

                    let logContent = '[' + log.timestamp + '] ' + log.file + '\n';
                    logContent += 'Size: ' + log.size_kb.toFixed(2) + ' KB\n';
                    logContent += 'Preview:\n' + log.preview + '\n';
                    logContent += '================================================================================\n\n';

                    logLine.textContent = logContent;
                    logOutput.appendChild(logLine);
                });
                showToast('System logs loaded successfully', 'success');
            } else {
                logOutput.innerHTML = '<span style="color: #ffb86c;">No logs available yet. Run pipelines to generate logs.</span>';
                showToast('No logs found', 'info');
            }
        }
    } catch (error) {
        console.error('Error loading system logs:', error);
        const logOutput = document.getElementById('logOutput');
        if (logOutput) {
            logOutput.innerHTML = '<span style="color: #ff5555;">Error loading logs: ' + error.message + '</span>';
        }
        showToast('Error: ' + error.message, 'error');
    }
}

// Load full log content (not just preview)
async function loadFullLogs() {
    showSpinner();
    try {
        const response = await fetch(API_BASE + '/ml/pipeline-logs?limit=1');
        const result = await response.json();

        const logOutput = document.getElementById('logOutput');
        if (logOutput) {
            logOutput.innerHTML = '';

            if (result.status === 'success' && result.data.data && result.data.data.length > 0) {
                const latestLog = result.data.data[0];

                const logLine = document.createElement('div');
                logLine.className = 'log-line';

                // Get full log content
                let fullContent = '[' + latestLog.timestamp + '] Latest Log File: ' + latestLog.file + '\n';
                fullContent += 'File Size: ' + latestLog.size_kb.toFixed(2) + ' KB\n';
                fullContent += '================================================================================\n\n';
                fullContent += latestLog.full_content || latestLog.preview + '\n\n';
                fullContent += '================================================================================\n';

                logLine.textContent = fullContent;
                logOutput.appendChild(logLine);

                showToast('Latest log file loaded successfully', 'success');
            } else {
                logOutput.innerHTML = '<span style="color: #ffb86c;">No logs available. Run the ML pipeline first to generate logs.</span>';
                showToast('No logs found', 'info');
            }
        }
    } catch (error) {
        console.error('Error loading full logs:', error);
        const logOutput = document.getElementById('logOutput');
        if (logOutput) {
            logOutput.innerHTML = '<span style="color: #ff5555;">Error loading logs: ' + error.message + '</span>';
        }
        showToast('Error: ' + error.message, 'error');
    }
    hideSpinner();
}

// Refresh logs in real-time
async function refreshLogs() {
    showSpinner();
    try {
        const response = await fetch(API_BASE + '/ml/pipeline-logs?limit=1');
        const result = await response.json();

        const logOutput = document.getElementById('logOutput');
        if (logOutput && result.status === 'success' && result.data.data && result.data.data.length > 0) {
            const latestLog = result.data.data[0];

            // Clear and reload
            logOutput.innerHTML = '';
            const logLine = document.createElement('div');
            logLine.className = 'log-line';

            let fullContent = '[' + latestLog.timestamp + '] ' + latestLog.file + '\n';
            fullContent += 'Size: ' + latestLog.size_kb.toFixed(2) + ' KB\n';
            fullContent += '================================================================================\n\n';
            fullContent += latestLog.full_content || latestLog.preview + '\n\n';
            fullContent += '================================================================================\n';

            logLine.textContent = fullContent;
            logOutput.appendChild(logLine);

            showToast('Logs refreshed - ' + new Date().toLocaleTimeString(), 'success');
        }
    } catch (error) {
        console.error('Error refreshing logs:', error);
        showToast('Error refreshing logs: ' + error.message, 'error');
    }
    hideSpinner();
}

// Search/Filter logs by text
function filterLogs() {
    const filterInput = document.getElementById('logFilter');
    const logOutput = document.getElementById('logOutput');

    if (!filterInput || !logOutput) return;

    const filter = filterInput.value.toLowerCase();
    const logLines = document.querySelectorAll('.log-line');

    if (logLines.length === 0) {
        showToast('No logs loaded yet. Click "Load Latest Log" first.', 'info');
        return;
    }

    let matchCount = 0;
    logLines.forEach(function(line) {
        const text = line.textContent.toLowerCase();
        if (filter === '' || text.indexOf(filter) > -1) {
            line.style.display = 'block';
            matchCount++;
        } else {
            line.style.display = 'none';
        }
    });

    if (filter === '') {
        showToast('Filter cleared - showing all logs', 'success');
    } else {
        showToast('Found ' + matchCount + ' matching log entries', 'success');
    }
}

function clearLogs() {
    const logOutput = document.getElementById('logOutput');
    if (logOutput) logOutput.innerHTML = '';
    const logFilter = document.getElementById('logFilter');
    if (logFilter) logFilter.value = '';
    showToast('Logs cleared', 'info');
}

// ============ UTILITIES ============
function showSpinner() {
    const spinner = document.getElementById('loadingSpinner');
    if (spinner) spinner.classList.remove('hidden');
}

function hideSpinner() {
    const spinner = document.getElementById('loadingSpinner');
    if (spinner) spinner.classList.add('hidden');
}

function showToast(message, type) {
    type = type || 'info';
    const container = document.getElementById('toastContainer');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = 'toast toast-' + type;

    const icon = type === 'success' ? '✓' : type === 'error' ? '✗' : 'ℹ';
    toast.innerHTML = '<span>' + icon + '</span><span>' + message + '</span>';

    container.appendChild(toast);

    setTimeout(function() {
        toast.style.animation = 'slideIn 0.3s ease reverse';
        setTimeout(function() { toast.remove(); }, 300);
    }, 3000);
}

function refreshData() {
    const activePage = document.querySelector('.page.active');
    if (activePage && activePage.id === 'overview-page') {
        loadDashboardOverview();
    }

    showToast('Data refreshed', 'success');
}

function toggleDarkMode() {
    document.body.classList.toggle('dark-mode');
    localStorage.setItem('darkMode', document.body.classList.contains('dark-mode'));
}

// Check for saved dark mode preference
if (localStorage.getItem('darkMode') === 'true') {
    document.body.classList.add('dark-mode');
}

// Responsive handling
window.addEventListener('resize', function() {
    if (window.innerWidth > 768) {
        const sidebar = document.querySelector('.sidebar');
        if (sidebar) sidebar.classList.remove('active');
    }
});

