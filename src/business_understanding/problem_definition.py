"""
Business Understanding and Problem Definition for Predictive Maintenance
Author: ramkumarjayakumar
Date: 2025-10-18
"""

from typing import Dict, List
import json
from datetime import datetime


class BusinessProblemDefinition:
    """
    Comprehensive business problem definition for predictive maintenance
    of turbofan engines using NASA C-MAPSS dataset
    """

    def __init__(self):
        self.problem_statement = self._define_problem_statement()
        self.kpis = self._define_kpis()
        self.business_impact = self._define_business_impact()
        self.cost_benefit = self._define_cost_benefit_framework()

    def _define_problem_statement(self) -> Dict:
        """Define the core business problem"""
        return {
            "title": "Predictive Maintenance for Turbofan Engines",
            "description": """
            Industrial equipment failures, particularly in turbofan engines, result in 
            significant operational costs, safety risks, and unplanned downtime. Traditional 
            reactive or time-based maintenance strategies are inefficient and costly.
            
            This project aims to develop a predictive maintenance system that can:
            1. Predict remaining useful life (RUL) of turbofan engines
            2. Identify early warning signs of potential failures
            3. Optimize maintenance scheduling to reduce costs
            4. Minimize unplanned downtime and safety incidents
            5. Extend equipment lifespan through proactive interventions
            """,
            "business_context": {
                "industry": "Aerospace and Aviation",
                "equipment_type": "Turbofan Engines",
                "dataset": "NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)",
                "data_sources": ["FD002 (53,000+ records)", "FD004 (61,000+ records)"],
                "sensor_data_points": 21,
                "operational_settings": 3,
                "total_engines_monitored": "259 engines (FD002: 260, FD004: 249 test engines)"
            },
            "objectives": [
                "Develop accurate RUL prediction models",
                "Reduce maintenance costs by 30%",
                "Decrease unplanned downtime by 40%",
                "Achieve 90%+ prediction accuracy",
                "Enable real-time monitoring and alerting"
            ],
            "stakeholders": [
                "Maintenance Teams",
                "Operations Managers",
                "Safety Officers",
                "Finance Department",
                "Engineering Teams"
            ]
        }

    def _define_kpis(self) -> Dict:
        """Define Key Performance Indicators"""
        return {
            "predictive_accuracy": {
                "metric": "Model Accuracy",
                "target": "≥ 90%",
                "description": "Percentage of correct RUL predictions within acceptable margin",
                "measurement": "Mean Absolute Error (MAE) on test set"
            },
            "precision": {
                "metric": "Precision Score",
                "target": "≥ 85%",
                "description": "Ratio of true positive predictions to total positive predictions",
                "measurement": "TP / (TP + FP)",
                "importance": "Minimize false alarms"
            },
            "recall": {
                "metric": "Recall/Sensitivity",
                "target": "≥ 90%",
                "description": "Ratio of correctly identified failures to total actual failures",
                "measurement": "TP / (TP + FN)",
                "importance": "Critical - missing a failure is costly and dangerous"
            },
            "f1_score": {
                "metric": "F1 Score",
                "target": "≥ 87%",
                "description": "Harmonic mean of precision and recall",
                "measurement": "2 * (Precision * Recall) / (Precision + Recall)"
            },
            "false_negative_rate": {
                "metric": "False Negative Rate",
                "target": "≤ 5%",
                "description": "Percentage of missed failure predictions",
                "measurement": "FN / (FN + TP)",
                "importance": "Critical safety metric"
            },
            "prediction_lead_time": {
                "metric": "Advance Warning Time",
                "target": "≥ 30 cycles",
                "description": "How far in advance failures are predicted",
                "measurement": "Average RUL prediction at failure threshold"
            },
            "maintenance_cost_reduction": {
                "metric": "Cost Savings",
                "target": "30% reduction",
                "description": "Percentage reduction in maintenance costs",
                "measurement": "Comparison with baseline reactive maintenance"
            },
            "downtime_reduction": {
                "metric": "Unplanned Downtime",
                "target": "40% reduction",
                "description": "Reduction in unexpected equipment failures",
                "measurement": "Hours of unplanned downtime per month"
            },
            "data_pipeline_performance": {
                "metric": "Pipeline Execution Time",
                "target": "< 5 minutes",
                "description": "Time to complete full data pipeline execution",
                "measurement": "End-to-end pipeline runtime"
            },
            "data_quality_score": {
                "metric": "Data Quality",
                "target": "≥ 95%",
                "description": "Percentage of data passing quality checks",
                "measurement": "Valid records / Total records"
            }
        }

    def _define_business_impact(self) -> Dict:
        """Define business impact metrics"""
        return {
            "financial_impact": {
                "cost_categories": {
                    "unplanned_downtime": {
                        "current_annual_cost": "$5,000,000",
                        "target_reduction": "40%",
                        "projected_savings": "$2,000,000"
                    },
                    "emergency_repairs": {
                        "current_annual_cost": "$3,000,000",
                        "target_reduction": "50%",
                        "projected_savings": "$1,500,000"
                    },
                    "spare_parts_inventory": {
                        "current_annual_cost": "$2,000,000",
                        "target_reduction": "20%",
                        "projected_savings": "$400,000"
                    },
                    "labor_overtime": {
                        "current_annual_cost": "$1,500,000",
                        "target_reduction": "30%",
                        "projected_savings": "$450,000"
                    }
                },
                "total_projected_annual_savings": "$4,350,000",
                "implementation_cost": "$500,000",
                "roi_timeline": "4-6 months",
                "5_year_roi": "4250%"
            },
            "operational_impact": {
                "equipment_availability": {
                    "current": "85%",
                    "target": "95%",
                    "improvement": "+10%"
                },
                "mean_time_between_failures": {
                    "current": "200 hours",
                    "target": "300 hours",
                    "improvement": "+50%"
                },
                "maintenance_efficiency": {
                    "current": "60%",
                    "target": "85%",
                    "improvement": "+25%"
                },
                "resource_utilization": {
                    "current": "70%",
                    "target": "90%",
                    "improvement": "+20%"
                }
            },
            "safety_impact": {
                "risk_reduction": "60% reduction in catastrophic failures",
                "safety_incidents": "80% reduction in maintenance-related incidents",
                "compliance": "100% regulatory compliance through proactive monitoring",
                "reputation": "Enhanced safety record and brand reputation"
            },
            "competitive_advantage": {
                "service_reliability": "Industry-leading uptime guarantees",
                "customer_satisfaction": "Improved customer trust and retention",
                "market_position": "Technology leadership in predictive maintenance",
                "innovation": "Foundation for AI-driven operations"
            }
        }

    def _define_cost_benefit_framework(self) -> Dict:
        """Define cost-benefit analysis framework"""
        return {
            "costs": {
                "implementation": {
                    "data_infrastructure": "$150,000",
                    "ml_development": "$200,000",
                    "aws_setup": "$50,000",
                    "training_documentation": "$50,000",
                    "testing_validation": "$50,000",
                    "total": "$500,000"
                },
                "ongoing_annual": {
                    "aws_services": "$50,000",
                    "maintenance_support": "$80,000",
                    "model_updates": "$40,000",
                    "staff_training": "$30,000",
                    "total": "$200,000"
                }
            },
            "benefits": {
                "year_1": {
                    "cost_savings": "$4,350,000",
                    "implementation_cost": "-$500,000",
                    "ongoing_cost": "-$200,000",
                    "net_benefit": "$3,650,000"
                },
                "year_2_5": {
                    "annual_cost_savings": "$4,350,000",
                    "annual_ongoing_cost": "-$200,000",
                    "annual_net_benefit": "$4,150,000",
                    "total_4_year_benefit": "$16,600,000"
                },
                "5_year_total": {
                    "total_savings": "$21,750,000",
                    "total_costs": "$1,300,000",
                    "net_benefit": "$20,450,000",
                    "roi": "1573%"
                }
            },
            "risk_analysis": {
                "low_risk": {
                    "description": "Technology well-established, proven ROI",
                    "probability": "High",
                    "mitigation": "Use proven ML frameworks and AWS services"
                },
                "data_quality_risk": {
                    "description": "Insufficient or poor quality sensor data",
                    "probability": "Medium",
                    "mitigation": "Comprehensive data validation and quality checks"
                },
                "adoption_risk": {
                    "description": "User resistance to new system",
                    "probability": "Low",
                    "mitigation": "Comprehensive training and change management"
                },
                "technical_risk": {
                    "description": "Model accuracy below targets",
                    "probability": "Low",
                    "mitigation": "Multiple model evaluation, continuous improvement"
                }
            },
            "success_criteria": {
                "critical": [
                    "Model accuracy ≥ 90%",
                    "False negative rate ≤ 5%",
                    "System uptime ≥ 99%",
                    "ROI achieved within 6 months"
                ],
                "important": [
                    "User adoption ≥ 80%",
                    "Data quality ≥ 95%",
                    "Pipeline execution < 5 minutes",
                    "Cost reduction ≥ 30%"
                ],
                "nice_to_have": [
                    "Real-time dashboard",
                    "Mobile alerts",
                    "Integration with existing systems",
                    "Automated reporting"
                ]
            }
        }

    def generate_report(self) -> str:
        """Generate comprehensive business problem definition report"""
        report = f"""
{'='*80}
PREDICTIVE MAINTENANCE BUSINESS PROBLEM DEFINITION
NASA C-MAPSS Turbofan Engine Dataset
{'='*80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Author: ramkumarjayakumar

{'-'*80}
1. PROBLEM STATEMENT
{'-'*80}
Title: {self.problem_statement['title']}

{self.problem_statement['description']}

Business Context:
  - Industry: {self.problem_statement['business_context']['industry']}
  - Equipment: {self.problem_statement['business_context']['equipment_type']}
  - Dataset: {self.problem_statement['business_context']['dataset']}
  - Data Sources: {', '.join(self.problem_statement['business_context']['data_sources'])}
  - Sensors: {self.problem_statement['business_context']['sensor_data_points']} sensor measurements
  - Engines: {self.problem_statement['business_context']['total_engines_monitored']}

Objectives:
"""
        for obj in self.problem_statement['objectives']:
            report += f"  • {obj}\n"

        report += f"""
{'-'*80}
2. KEY PERFORMANCE INDICATORS (KPIs)
{'-'*80}
"""
        for kpi_name, kpi_data in self.kpis.items():
            report += f"\n{kpi_data['metric']}: {kpi_data['target']}\n"
            report += f"  Description: {kpi_data['description']}\n"

        report += f"""
{'-'*80}
3. BUSINESS IMPACT
{'-'*80}

Financial Impact:
  Total Projected Annual Savings: {self.business_impact['financial_impact']['total_projected_annual_savings']}
  Implementation Cost: {self.business_impact['financial_impact']['implementation_cost']}
  ROI Timeline: {self.business_impact['financial_impact']['roi_timeline']}
  5-Year ROI: {self.business_impact['financial_impact']['5_year_roi']}

Operational Impact:
  Equipment Availability: {self.business_impact['operational_impact']['equipment_availability']['current']} → {self.business_impact['operational_impact']['equipment_availability']['target']}
  MTBF Improvement: {self.business_impact['operational_impact']['mean_time_between_failures']['improvement']}
  Maintenance Efficiency: {self.business_impact['operational_impact']['maintenance_efficiency']['improvement']}

{'-'*80}
4. COST-BENEFIT ANALYSIS
{'-'*80}

5-Year Financial Summary:
  Total Savings: {self.cost_benefit['benefits']['5_year_total']['total_savings']}
  Total Costs: {self.cost_benefit['benefits']['5_year_total']['total_costs']}
  Net Benefit: {self.cost_benefit['benefits']['5_year_total']['net_benefit']}
  ROI: {self.cost_benefit['benefits']['5_year_total']['roi']}

Year 1 Net Benefit: {self.cost_benefit['benefits']['year_1']['net_benefit']}

{'='*80}
END OF BUSINESS PROBLEM DEFINITION
{'='*80}
"""
        return report

    def save_report(self, filepath: str):
        """Save business problem definition report to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.generate_report())

    def export_json(self, filepath: str):
        """Export business problem definition as JSON"""
        data = {
            "problem_statement": self.problem_statement,
            "kpis": self.kpis,
            "business_impact": self.business_impact,
            "cost_benefit": self.cost_benefit,
            "generated_date": datetime.now().isoformat()
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # Create business problem definition
    business_def = BusinessProblemDefinition()

    # Generate and print report
    print(business_def.generate_report())

    # Save to files
    from pathlib import Path
    base_dir = Path(__file__).resolve().parent.parent.parent
    docs_dir = base_dir / "docs"
    docs_dir.mkdir(exist_ok=True)

    business_def.save_report(docs_dir / "business_problem_definition.txt")
    business_def.export_json(docs_dir / "business_problem_definition.json")

    print("\nReports saved to docs/ directory")

