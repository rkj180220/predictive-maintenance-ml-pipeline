"""
Dashboard Generator for Predictive Maintenance Pipeline
Author: ramkumarjayakumar
Date: 2025-10-18
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class PipelineDashboard:
    """
    Generate real-time dashboard for pipeline monitoring
    Creates HTML dashboards with metrics, status, and visualizations
    """

    def __init__(self, output_dir: Path):
        """
        Initialize dashboard generator

        Args:
            output_dir: Directory to save dashboard files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_dashboard(self, monitor_data: Dict, scheduler_status: Dict,
                          output_filename: str = "dashboard.html"):
        """
        Generate comprehensive dashboard

        Args:
            monitor_data: Monitoring data dictionary
            scheduler_status: Scheduler status dictionary
            output_filename: Output HTML filename
        """
        logger.info("Generating pipeline dashboard...")

        # Create HTML dashboard
        html_content = self._create_html_dashboard(monitor_data, scheduler_status)

        # Save to file
        output_path = self.output_dir / output_filename
        with open(output_path, 'w') as f:
            f.write(html_content)

        logger.info(f"Dashboard saved to {output_path}")

        return output_path

    def _create_html_dashboard(self, monitor_data: Dict, scheduler_status: Dict) -> str:
        """Create HTML dashboard content"""

        # Get performance summary
        perf_summary = monitor_data.get('performance_summary', {})

        # Create status indicator
        status_color = "green" if scheduler_status.get('is_running') else "red"
        status_text = "RUNNING" if scheduler_status.get('is_running') else "STOPPED"

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="30">
    <title>Predictive Maintenance Pipeline Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        
        .header h1 {{
            color: #667eea;
            margin-bottom: 10px;
        }}
        
        .status-badge {{
            display: inline-block;
            padding: 8px 16px;
            background: {status_color};
            color: white;
            border-radius: 20px;
            font-weight: bold;
            font-size: 14px;
        }}
        
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .card {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .card h2 {{
            color: #667eea;
            margin-bottom: 15px;
            font-size: 18px;
            border-bottom: 2px solid #f0f0f0;
            padding-bottom: 10px;
        }}
        
        .metric {{
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #f0f0f0;
        }}
        
        .metric:last-child {{
            border-bottom: none;
        }}
        
        .metric-label {{
            color: #666;
            font-weight: 500;
        }}
        
        .metric-value {{
            color: #333;
            font-weight: bold;
        }}
        
        .success {{
            color: #10b981;
        }}
        
        .warning {{
            color: #f59e0b;
        }}
        
        .error {{
            color: #ef4444;
        }}
        
        .timestamp {{
            color: #999;
            font-size: 12px;
            text-align: center;
            margin-top: 20px;
        }}
        
        .alert {{
            background: #fef3c7;
            border-left: 4px solid #f59e0b;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        th, td {{
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #f0f0f0;
        }}
        
        th {{
            background: #f9fafb;
            color: #667eea;
            font-weight: 600;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔧 Predictive Maintenance Pipeline Dashboard</h1>
            <p>Real-time monitoring and analytics for NASA C-MAPSS turbofan engine analysis</p>
            <div style="margin-top: 15px;">
                <span class="status-badge">{status_text}</span>
                <span style="margin-left: 15px; color: #666;">
                    Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </span>
            </div>
        </div>
        
        <div class="grid">
            <div class="card">
                <h2>📊 Scheduler Status</h2>
                <div class="metric">
                    <span class="metric-label">Status</span>
                    <span class="metric-value {'success' if scheduler_status.get('is_running') else 'error'}">
                        {status_text}
                    </span>
                </div>
                <div class="metric">
                    <span class="metric-label">Execution Count</span>
                    <span class="metric-value">{scheduler_status.get('execution_count', 0)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Schedule Interval</span>
                    <span class="metric-value">{scheduler_status.get('schedule_interval_minutes', 0)} minutes</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Last Execution</span>
                    <span class="metric-value">{scheduler_status.get('last_execution_time', 'N/A')}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Next Run</span>
                    <span class="metric-value">{scheduler_status.get('next_run', 'N/A')}</span>
                </div>
            </div>
            
            <div class="card">
                <h2>⚡ Performance Metrics</h2>
                <div class="metric">
                    <span class="metric-label">Total Executions</span>
                    <span class="metric-value">{perf_summary.get('total_executions', 0)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Success Rate</span>
                    <span class="metric-value success">
                        {perf_summary.get('success_rate', 0) * 100:.1f}%
                    </span>
                </div>
                <div class="metric">
                    <span class="metric-label">Avg Duration</span>
                    <span class="metric-value">
                        {perf_summary.get('average_duration_seconds', 0):.2f}s
                    </span>
                </div>
                <div class="metric">
                    <span class="metric-label">Min Duration</span>
                    <span class="metric-value">{perf_summary.get('min_duration_seconds', 0):.2f}s</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Max Duration</span>
                    <span class="metric-value">{perf_summary.get('max_duration_seconds', 0):.2f}s</span>
                </div>
            </div>
            
            <div class="card">
                <h2>✅ Execution Summary</h2>
                <div class="metric">
                    <span class="metric-label">Successful</span>
                    <span class="metric-value success">
                        {perf_summary.get('successful_executions', 0)}
                    </span>
                </div>
                <div class="metric">
                    <span class="metric-label">Failed</span>
                    <span class="metric-value error">
                        {perf_summary.get('failed_executions', 0)}
                    </span>
                </div>
                <div class="metric">
                    <span class="metric-label">Total Alerts</span>
                    <span class="metric-value warning">
                        {perf_summary.get('total_alerts', 0)}
                    </span>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>📈 Recent Executions</h2>
            <table>
                <thead>
                    <tr>
                        <th>Execution ID</th>
                        <th>Status</th>
                        <th>Duration</th>
                        <th>Start Time</th>
                        <th>Stages</th>
                    </tr>
                </thead>
                <tbody>
"""

        # Add recent executions
        executions = monitor_data.get('executions', [])
        for execution in executions[-10:]:
            status_class = 'success' if execution.get('status') == 'SUCCESS' else 'error'
            html += f"""
                    <tr>
                        <td>{execution.get('execution_id', 'N/A')}</td>
                        <td><span class="{status_class}">{execution.get('status', 'N/A')}</span></td>
                        <td>{execution.get('duration_seconds', 0):.2f}s</td>
                        <td>{execution.get('start_time', 'N/A')}</td>
                        <td>{len(execution.get('stages', []))}</td>
                    </tr>
"""

        html += """
                </tbody>
            </table>
        </div>
        
        <div class="timestamp">
            Auto-refresh every 30 seconds | Generated by Predictive Maintenance ML Pipeline
        </div>
    </div>
</body>
</html>
"""

        return html

    def generate_metrics_dashboard(self, metrics: Dict, output_filename: str = "metrics.html"):
        """
        Generate metrics visualization dashboard

        Args:
            metrics: Metrics dictionary
            output_filename: Output filename
        """
        logger.info("Generating metrics dashboard...")

        # Create plotly figures
        fig = self._create_metrics_plots(metrics)

        # Save to HTML
        output_path = self.output_dir / output_filename
        fig.write_html(output_path)

        logger.info(f"Metrics dashboard saved to {output_path}")

        return output_path

    def _create_metrics_plots(self, metrics: Dict) -> go.Figure:
        """Create plotly metrics visualizations"""

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Execution Duration Over Time',
                          'Success Rate',
                          'Stage Performance',
                          'Alert Distribution')
        )

        # Add placeholder data (will be populated with actual metrics)
        fig.add_trace(
            go.Scatter(x=[1, 2, 3], y=[10, 15, 12], name="Duration"),
            row=1, col=1
        )

        fig.update_layout(
            height=800,
            title_text="Pipeline Metrics Dashboard",
            showlegend=True
        )

        return fig


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    from pathlib import Path

    # Test dashboard generation
    dashboard = PipelineDashboard(Path("./dashboards"))

    test_monitor_data = {
        'performance_summary': {
            'total_executions': 10,
            'successful_executions': 9,
            'failed_executions': 1,
            'success_rate': 0.9,
            'average_duration_seconds': 45.5,
            'min_duration_seconds': 40.2,
            'max_duration_seconds': 52.3,
            'total_alerts': 3
        },
        'executions': []
    }

    test_scheduler_status = {
        'is_running': True,
        'execution_count': 10,
        'schedule_interval_minutes': 2,
        'last_execution_time': datetime.now().isoformat(),
        'next_run': 'In 2 minutes'
    }

    dashboard.generate_dashboard(test_monitor_data, test_scheduler_status)
    print("Dashboard generated successfully")

