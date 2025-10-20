"""
Visualization Module for NASA C-MAPSS Dataset
Author: ramkumarjayakumar
Date: 2025-10-18
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class DataVisualizer:
    """
    Comprehensive visualization for NASA C-MAPSS dataset
    Creates static and interactive plots for EDA
    """

    def __init__(self, config: Dict):
        """
        Initialize visualizer

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.figure_size = config.get('figure_size', (12, 8))
        self.color_palette = config.get('color_palette', 'viridis')

        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette(self.color_palette)

    def create_correlation_heatmap(self, df: pd.DataFrame, output_path: Path,
                                   top_n: int = 21) -> None:
        """
        Create correlation heatmap for sensor data

        Args:
            df: DataFrame with features
            output_path: Path to save figure
            top_n: Number of features to include
        """
        logger.info("Creating correlation heatmap...")

        # Get sensor columns
        sensor_cols = [col for col in df.columns if col.startswith('sensor_')][:top_n]

        if len(sensor_cols) < 2:
            logger.warning("Not enough sensor columns for correlation heatmap")
            return

        # Calculate correlation matrix
        corr_matrix = df[sensor_cols].corr()

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 12))

        # Create heatmap
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )

        plt.title('Sensor Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()

        # Save figure
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Correlation heatmap saved to {output_path}")

    def create_rul_distribution(self, df: pd.DataFrame, output_path: Path) -> None:
        """
        Create RUL distribution plots

        Args:
            df: DataFrame with RUL column
            output_path: Path to save figure
        """
        logger.info("Creating RUL distribution plots...")

        if 'RUL' not in df.columns:
            logger.warning("RUL column not found")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Histogram
        axes[0, 0].hist(df['RUL'], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('RUL (cycles)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('RUL Distribution')
        axes[0, 0].grid(True, alpha=0.3)

        # Box plot
        axes[0, 1].boxplot(df['RUL'], vert=True)
        axes[0, 1].set_ylabel('RUL (cycles)')
        axes[0, 1].set_title('RUL Box Plot')
        axes[0, 1].grid(True, alpha=0.3)

        # KDE plot
        df['RUL'].plot(kind='kde', ax=axes[1, 0])
        axes[1, 0].set_xlabel('RUL (cycles)')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('RUL Density Plot')
        axes[1, 0].grid(True, alpha=0.3)

        # Cumulative distribution
        sorted_rul = np.sort(df['RUL'])
        y = np.arange(1, len(sorted_rul) + 1) / len(sorted_rul)
        axes[1, 1].plot(sorted_rul, y)
        axes[1, 1].set_xlabel('RUL (cycles)')
        axes[1, 1].set_ylabel('Cumulative Probability')
        axes[1, 1].set_title('RUL Cumulative Distribution')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"RUL distribution plots saved to {output_path}")

    def create_sensor_distributions(self, df: pd.DataFrame, output_path: Path,
                                    sensors: List[str] = None) -> None:
        """
        Create distribution plots for sensors

        Args:
            df: DataFrame with sensor data
            output_path: Path to save figure
            sensors: List of sensor columns to plot
        """
        logger.info("Creating sensor distribution plots...")

        if sensors is None:
            sensors = [col for col in df.columns if col.startswith('sensor_')][:12]

        n_sensors = len(sensors)
        if n_sensors == 0:
            logger.warning("No sensor columns found")
            return

        n_cols = 4
        n_rows = (n_sensors + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
        axes = axes.flatten() if n_sensors > 1 else [axes]

        for idx, sensor in enumerate(sensors):
            if sensor in df.columns:
                axes[idx].hist(df[sensor].dropna(), bins=30, edgecolor='black', alpha=0.7)
                axes[idx].set_xlabel(sensor)
                axes[idx].set_ylabel('Frequency')
                axes[idx].set_title(f'{sensor} Distribution')
                axes[idx].grid(True, alpha=0.3)

        # Hide extra subplots
        for idx in range(n_sensors, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Sensor distribution plots saved to {output_path}")

    def create_degradation_patterns(self, df: pd.DataFrame, output_path: Path,
                                    n_engines: int = 5) -> None:
        """
        Create degradation pattern visualizations

        Args:
            df: DataFrame with engine data
            output_path: Path to save figure
            n_engines: Number of engines to plot
        """
        logger.info("Creating degradation pattern plots...")

        if 'unit_id' not in df.columns or 'time_cycles' not in df.columns:
            logger.warning("Required columns not found")
            return

        # Select sample engines
        engine_ids = df['unit_id'].unique()[:n_engines]

        # Get key sensors
        sensor_cols = [col for col in df.columns if col.startswith('sensor_')][:4]

        fig, axes = plt.subplots(len(sensor_cols), 1, figsize=(14, len(sensor_cols) * 3))
        if len(sensor_cols) == 1:
            axes = [axes]

        for idx, sensor in enumerate(sensor_cols):
            for engine_id in engine_ids:
                engine_data = df[df['unit_id'] == engine_id].sort_values('time_cycles')
                axes[idx].plot(
                    engine_data['time_cycles'],
                    engine_data[sensor],
                    label=f'Engine {engine_id}',
                    alpha=0.7
                )

            axes[idx].set_xlabel('Time Cycles')
            axes[idx].set_ylabel(sensor)
            axes[idx].set_title(f'{sensor} Degradation Over Time')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Degradation pattern plots saved to {output_path}")

    def create_rul_vs_sensors(self, df: pd.DataFrame, output_path: Path,
                              top_sensors: List[str] = None) -> None:
        """
        Create scatter plots of RUL vs sensor values

        Args:
            df: DataFrame with RUL and sensor data
            output_path: Path to save figure
            top_sensors: List of sensors to plot
        """
        logger.info("Creating RUL vs Sensor plots...")

        if 'RUL' not in df.columns:
            logger.warning("RUL column not found")
            return

        if top_sensors is None:
            top_sensors = [col for col in df.columns if col.startswith('sensor_')][:8]

        n_sensors = len(top_sensors)
        n_cols = 4
        n_rows = (n_sensors + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
        axes = axes.flatten() if n_sensors > 1 else [axes]

        for idx, sensor in enumerate(top_sensors):
            if sensor in df.columns:
                # Sample data for faster plotting
                sample_df = df.sample(min(10000, len(df)))
                axes[idx].scatter(
                    sample_df[sensor],
                    sample_df['RUL'],
                    alpha=0.3,
                    s=1
                )
                axes[idx].set_xlabel(sensor)
                axes[idx].set_ylabel('RUL')
                axes[idx].set_title(f'RUL vs {sensor}')
                axes[idx].grid(True, alpha=0.3)

        # Hide extra subplots
        for idx in range(n_sensors, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"RUL vs Sensor plots saved to {output_path}")

    def create_interactive_degradation(self, df: pd.DataFrame, output_path: Path,
                                       engine_id: int = None) -> None:
        """
        Create interactive degradation plot using Plotly

        Args:
            df: DataFrame with engine data
            output_path: Path to save HTML figure
            engine_id: Specific engine to plot
        """
        logger.info("Creating interactive degradation plot...")

        if 'unit_id' not in df.columns or 'time_cycles' not in df.columns:
            logger.warning("Required columns not found")
            return

        if engine_id is None:
            engine_id = df['unit_id'].iloc[0]

        engine_data = df[df['unit_id'] == engine_id].sort_values('time_cycles')

        # Get sensor columns
        sensor_cols = [col for col in df.columns if col.startswith('sensor_')][:6]

        # Create subplots
        fig = make_subplots(
            rows=len(sensor_cols), cols=1,
            subplot_titles=[f'{sensor}' for sensor in sensor_cols],
            vertical_spacing=0.05
        )

        for idx, sensor in enumerate(sensor_cols, 1):
            fig.add_trace(
                go.Scatter(
                    x=engine_data['time_cycles'],
                    y=engine_data[sensor],
                    mode='lines',
                    name=sensor,
                    line=dict(width=2)
                ),
                row=idx, col=1
            )

        fig.update_layout(
            height=300 * len(sensor_cols),
            title_text=f"Engine {engine_id} Degradation Patterns",
            showlegend=False
        )

        fig.update_xaxes(title_text="Time Cycles", row=len(sensor_cols), col=1)

        # Save as HTML
        fig.write_html(output_path)

        logger.info(f"Interactive degradation plot saved to {output_path}")

    def create_feature_correlation_with_rul(self, df: pd.DataFrame, output_path: Path,
                                            top_n: int = 15) -> None:
        """
        Create bar plot of feature correlations with RUL

        Args:
            df: DataFrame with features and RUL
            output_path: Path to save figure
            top_n: Number of top features to show
        """
        logger.info("Creating feature correlation with RUL plot...")

        if 'RUL' not in df.columns:
            logger.warning("RUL column not found")
            return

        # Calculate correlations
        exclude_cols = ['unit_id', 'time_cycles', 'RUL', 'failure_label', 'dataset']
        feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                       if col not in exclude_cols]

        correlations = {}
        for col in feature_cols:
            corr = df[col].corr(df['RUL'])
            if not np.isnan(corr):
                correlations[col] = corr

        # Sort by absolute correlation
        sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]

        features = [item[0] for item in sorted_corrs]
        corr_values = [item[1] for item in sorted_corrs]

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))

        colors = ['green' if x > 0 else 'red' for x in corr_values]
        bars = ax.barh(range(len(features)), corr_values, color=colors, alpha=0.7)

        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('Correlation with RUL')
        ax.set_title(f'Top {top_n} Features Correlated with RUL', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Feature correlation plot saved to {output_path}")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    from src.config.settings import EDA_CONFIG

    visualizer = DataVisualizer(EDA_CONFIG)
    print("Data Visualizer initialized successfully")

