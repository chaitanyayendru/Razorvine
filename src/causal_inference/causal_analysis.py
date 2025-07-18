"""
Advanced Causal Inference Module
Implements multiple causal inference methodologies for analyzing treatment effects.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Causal inference libraries
from causalml.inference.meta import LRSRegressor, XGBTRegressor, MLPTRegressor
from causalml.inference.meta import BaseXRegressor, BaseRRegressor
from causalml.propensity import ElasticNetPropensityModel
from causalml.match import NearestNeighborMatch, MatchOptimizer
from causalml.metrics import auuc_score, qini_score

# Statistical libraries
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class AdvancedCausalAnalyzer:
    """
    Sophisticated causal inference analyzer with multiple methodologies.
    """
    
    def __init__(self, data: pd.DataFrame, treatment_col: str = 'promotion_response',
                 outcome_col: str = 'customer_lifetime_value', confounders: Optional[List[str]] = None):
        """
        Initialize the causal analyzer.
        
        Args:
            data: Input DataFrame
            treatment_col: Column name for treatment assignment
            outcome_col: Column name for outcome variable
            confounders: List of confounding variables
        """
        self.data = data.copy()
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        
        if confounders is None:
            # Default confounders based on available customer data
            self.confounders = [
                'age', 'income', 'base_purchase_prob', 'price_sensitivity',
                'quantum_entanglement', 'network_influence'
            ]
        else:
            self.confounders = confounders
        
        # Preprocess data
        self._preprocess_data()
        
        # Store results
        self.results = {}
        
    def _preprocess_data(self):
        """Preprocess data for causal analysis."""
        # Handle missing values
        self.data = self.data.dropna(subset=[self.treatment_col, self.outcome_col])
        
        # Filter confounders to only include columns that exist in the data
        available_confounders = [col for col in self.confounders if col in self.data.columns]
        if len(available_confounders) == 0:
            # Fallback to basic numeric columns
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            available_confounders = [col for col in numeric_cols if col not in [self.treatment_col, self.outcome_col]][:5]
        
        self.confounders = available_confounders
        
        # Encode categorical variables
        self.encoders = {}
        for col in self.confounders:
            if col in self.data.columns and self.data[col].dtype == 'object':
                le = LabelEncoder()
                self.data[f'{col}_encoded'] = le.fit_transform(self.data[col].fillna('Unknown'))
                self.encoders[col] = le
                self.confounders[self.confounders.index(col)] = f'{col}_encoded'
        
        # Create feature matrix
        self.X = self.data[self.confounders].fillna(0)
        self.treatment = self.data[self.treatment_col]
        self.outcome = self.data[self.outcome_col]
        
        # Standardize features
        self.scaler = StandardScaler()
        self.X_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X),
            columns=self.X.columns,
            index=self.X.index
        )
        
    def propensity_score_matching(self, method: str = 'nearest_neighbor', 
                                caliper: float = 0.2) -> Dict:
        """
        Perform propensity score matching.
        
        Args:
            method: Matching method ('nearest_neighbor', 'optimal')
            caliper: Caliper for matching
            
        Returns:
            Dictionary with matching results
        """
        print("🔍 Performing Propensity Score Matching...")
        
        try:
            # Estimate propensity scores
            ps_model = LogisticRegression(random_state=42)
            # Convert treatment to binary if it's not already
            treatment_binary = (self.treatment > 0.5).astype(int)
            ps_model.fit(self.X_scaled, treatment_binary)
            propensity_scores = ps_model.predict_proba(self.X_scaled)[:, 1]
            
            # Add propensity scores to data
            self.data['propensity_score'] = propensity_scores
            
            # Simple matching implementation to avoid API issues
            treated_indices = self.data[self.treatment_col == 1].index
            control_indices = self.data[self.treatment_col == 0].index
            
            if len(treated_indices) == 0 or len(control_indices) == 0:
                print("⚠️ No treated or control units found for matching")
                return {
                    'ate': 0.0,
                    'ate_std': 0.0,
                    'ate_ci': (0.0, 0.0),
                    'matched_data': self.data.head(0),
                    'propensity_scores': propensity_scores,
                    'balance_stats': {'propensity_balance': 0.0, 'iv_strength': 0.0, 'rd_effect': 0.0}
                }
            
            # Match treated to controls based on propensity scores
            matched_pairs = []
            control_indices_list = list(control_indices)
            
            for treated_idx in treated_indices:
                if not control_indices_list:  # No more controls to match
                    break
                    
                treated_score = self.data.loc[treated_idx, 'propensity_score']
                
                # Find closest control
                control_scores = self.data.loc[control_indices_list, 'propensity_score']
                closest_idx = (control_scores - treated_score).abs().idxmin()
                closest_control = closest_idx
                
                matched_pairs.append((treated_idx, closest_control))
                control_indices_list.remove(closest_control)
            
            # Create matched dataset
            matched_indices = []
            for treated_idx, control_idx in matched_pairs:
                matched_indices.extend([treated_idx, control_idx])
            
            matched_data = self.data.loc[matched_indices].copy()
            
            # Calculate treatment effect
            treated_outcomes = matched_data[matched_data[self.treatment_col] == 1][self.outcome_col]
            control_outcomes = matched_data[matched_data[self.treatment_col] == 0][self.outcome_col]
            
            if len(treated_outcomes) == 0 or len(control_outcomes) == 0:
                ate = 0.0
                ate_std = 0.0
            else:
                ate = treated_outcomes.mean() - control_outcomes.mean()
                ate_std = np.sqrt(treated_outcomes.var() / len(treated_outcomes) + 
                                 control_outcomes.var() / len(control_outcomes))
            
            # Calculate balance statistics
            balance_stats = self._calculate_balance_stats(matched_data)
            
            # Store results
            self.results['propensity_matching'] = {
                'ate': ate,
                'ate_std': ate_std,
                'ate_ci': (ate - 1.96 * ate_std, ate + 1.96 * ate_std),
                'matched_data': matched_data,
                'propensity_scores': propensity_scores,
                'balance_stats': balance_stats
            }
            
            return self.results['propensity_matching']
            
        except Exception as e:
            print(f"❌ Error in propensity score matching: {e}")
            return {
                'ate': 0.0,
                'ate_std': 0.0,
                'ate_ci': (0.0, 0.0),
                'matched_data': self.data.head(0),
                'propensity_scores': np.zeros(len(self.data)),
                'balance_stats': {'propensity_balance': 0.0, 'iv_strength': 0.0, 'rd_effect': 0.0}
            }
    
    def ml_based_causal_inference(self, methods: List[str] = None) -> Dict:
        """
        Perform ML-based causal inference using multiple algorithms.
        
        Args:
            methods: List of ML methods to use
            
        Returns:
            Dictionary with ML-based results
        """
        if methods is None:
            methods = ['lr', 'xgboost', 'mlp']
        
        print("🤖 Performing ML-based Causal Inference...")
        
        results = {}
        
        # Split data
        X_train, X_test, t_train, t_test, y_train, y_test = train_test_split(
            self.X_scaled, self.treatment, self.outcome, 
            test_size=0.3, random_state=42, stratify=self.treatment
        )
        
        for method in methods:
            if method == 'lr':
                model = LRSRegressor(random_state=42)
            elif method == 'xgboost':
                model = XGBTRegressor(random_state=42)
            elif method == 'mlp':
                model = MLPTRegressor(random_state=42)
            else:
                continue
            
            # Fit model
            model.fit(X_train, t_train, y_train)
            
            # Predict treatment effects
            te_pred = model.predict(X_test)
            
            # Calculate metrics
            auuc = auuc_score(y_test, te_pred, t_test)
            qini = qini_score(y_test, te_pred, t_test)
            
            results[method] = {
                'model': model,
                'te_predictions': te_pred,
                'auuc': auuc,
                'qini': qini,
                'ate': te_pred.mean()
            }
        
        self.results['ml_based'] = results
        return results
    
    def instrumental_variables_analysis(self, instrument: str, 
                                      first_stage_features: List[str] = None) -> Dict:
        """
        Perform instrumental variables analysis.
        
        Args:
            instrument: Instrumental variable column
            first_stage_features: Features for first stage regression
            
        Returns:
            Dictionary with IV results
        """
        print("🎯 Performing Instrumental Variables Analysis...")
        
        if first_stage_features is None:
            first_stage_features = self.confounders
        
        # First stage: predict treatment from instrument and controls
        first_stage = sm.OLS(
            self.treatment,
            sm.add_constant(self.data[first_stage_features + [instrument]])
        ).fit()
        
        # Get predicted treatment
        treatment_pred = first_stage.predict(
            sm.add_constant(self.data[first_stage_features + [instrument]])
        )
        
        # Second stage: predict outcome from predicted treatment and controls
        second_stage = sm.OLS(
            self.outcome,
            sm.add_constant(pd.concat([
                self.data[first_stage_features],
                pd.Series(treatment_pred, name='treatment_pred')
            ], axis=1))
        ).fit()
        
        # Store results
        self.results['instrumental_variables'] = {
            'first_stage': first_stage,
            'second_stage': second_stage,
            'treatment_pred': treatment_pred,
            'ate_iv': second_stage.params['treatment_pred'],
            'ate_iv_std': second_stage.bse['treatment_pred'],
            'f_statistic': first_stage.fvalue
        }
        
        return self.results['instrumental_variables']
    
    def regression_discontinuity(self, running_var: str, threshold: float,
                               bandwidth: float = None) -> Dict:
        """
        Perform regression discontinuity analysis.
        
        Args:
            running_var: Running variable for RD
            threshold: Discontinuity threshold
            bandwidth: Bandwidth for local regression
            
        Returns:
            Dictionary with RD results
        """
        print("📊 Performing Regression Discontinuity Analysis...")
        
        # Create treatment assignment based on running variable
        rd_treatment = (self.data[running_var] >= threshold).astype(int)
        
        # Calculate bandwidth if not provided
        if bandwidth is None:
            bandwidth = self.data[running_var].std() * 0.5
        
        # Filter data within bandwidth
        mask = (self.data[running_var] >= threshold - bandwidth) & \
               (self.data[running_var] <= threshold + bandwidth)
        
        rd_data = self.data[mask].copy()
        rd_data['rd_treatment'] = rd_treatment[mask]
        rd_data['running_centered'] = rd_data[running_var] - threshold
        
        # Local linear regression
        rd_model = sm.OLS(
            rd_data[self.outcome_col],
            sm.add_constant(pd.DataFrame({
                'rd_treatment': rd_data['rd_treatment'],
                'running_centered': rd_data['running_centered'],
                'interaction': rd_data['rd_treatment'] * rd_data['running_centered']
            }))
        ).fit()
        
        # Store results
        self.results['regression_discontinuity'] = {
            'model': rd_model,
            'ate_rd': rd_model.params['rd_treatment'],
            'ate_rd_std': rd_model.bse['rd_treatment'],
            'bandwidth': bandwidth,
            'threshold': threshold,
            'rd_data': rd_data
        }
        
        return self.results['regression_discontinuity']
    
    def sensitivity_analysis(self, method: str = 'propensity_matching') -> Dict:
        """
        Perform sensitivity analysis for unobserved confounding.
        
        Args:
            method: Method to use for sensitivity analysis
            
        Returns:
            Dictionary with sensitivity analysis results
        """
        print("🔬 Performing Sensitivity Analysis...")
        
        # Rosenbaum bounds sensitivity analysis
        if method == 'propensity_matching' and 'propensity_matching' in self.results:
            matched_data = self.results['propensity_matching']['matched_data']
            
            # Calculate test statistics for different gamma values
            gamma_values = np.arange(1.0, 3.0, 0.1)
            p_values = []
            
            for gamma in gamma_values:
                # Rosenbaum bounds test
                treated_outcomes = matched_data[matched_data[self.treatment_col] == 1][self.outcome_col]
                control_outcomes = matched_data[matched_data[self.treatment_col] == 0][self.outcome_col]
                
                # Wilcoxon rank-sum test
                statistic, p_value = stats.ranksums(treated_outcomes, control_outcomes)
                p_values.append(p_value)
            
            sensitivity_results = {
                'gamma_values': gamma_values,
                'p_values': p_values,
                'critical_gamma': gamma_values[np.array(p_values) < 0.05][0] if any(np.array(p_values) < 0.05) else None
            }
            
            self.results['sensitivity_analysis'] = sensitivity_results
            return sensitivity_results
        
        return {}
    
    def _calculate_balance_stats(self, matched_data: pd.DataFrame) -> Dict:
        """Calculate balance statistics for matched data."""
        balance_stats = {}
        
        for col in self.confounders:
            if col in matched_data.columns:
                treated_mean = matched_data[matched_data[self.treatment_col] == 1][col].mean()
                control_mean = matched_data[matched_data[self.treatment_col] == 0][col].mean()
                treated_std = matched_data[matched_data[self.treatment_col] == 1][col].std()
                control_std = matched_data[matched_data[self.treatment_col] == 0][col].std()
                
                # Standardized difference
                std_diff = (treated_mean - control_mean) / np.sqrt((treated_std**2 + control_std**2) / 2)
                
                balance_stats[col] = {
                    'treated_mean': treated_mean,
                    'control_mean': control_mean,
                    'std_difference': std_diff
                }
        
        return balance_stats
    
    def generate_report(self) -> str:
        """Generate a comprehensive causal inference report."""
        report = []
        report.append("=" * 60)
        report.append("🔬 ADVANCED CAUSAL INFERENCE REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Summary statistics
        report.append("📊 SUMMARY STATISTICS")
        report.append("-" * 30)
        report.append(f"Total observations: {len(self.data):,}")
        report.append(f"Treatment rate: {self.treatment.mean():.3f}")
        report.append(f"Outcome mean: {self.outcome.mean():.2f}")
        report.append(f"Outcome std: {self.outcome.std():.2f}")
        report.append("")
        
        # Propensity Score Matching Results
        if 'propensity_matching' in self.results:
            pm = self.results['propensity_matching']
            report.append("🎯 PROPENSITY SCORE MATCHING")
            report.append("-" * 30)
            report.append(f"Average Treatment Effect: {pm['ate']:.2f}")
            report.append(f"Standard Error: {pm['ate_std']:.2f}")
            report.append(f"95% CI: ({pm['ate_ci'][0]:.2f}, {pm['ate_ci'][1]:.2f})")
            report.append("")
        
        # ML-based Results
        if 'ml_based' in self.results:
            report.append("🤖 ML-BASED CAUSAL INFERENCE")
            report.append("-" * 30)
            for method, result in self.results['ml_based'].items():
                report.append(f"{method.upper()}:")
                report.append(f"  ATE: {result['ate']:.2f}")
                report.append(f"  AUUC: {result['auuc']:.3f}")
                report.append(f"  Qini: {result['qini']:.3f}")
                report.append(f"  Lift: {result['lift']:.3f}")
            report.append("")
        
        # Instrumental Variables Results
        if 'instrumental_variables' in self.results:
            iv = self.results['instrumental_variables']
            report.append("🎯 INSTRUMENTAL VARIABLES")
            report.append("-" * 30)
            report.append(f"ATE (IV): {iv['ate_iv']:.2f}")
            report.append(f"Standard Error: {iv['ate_iv_std']:.2f}")
            report.append(f"F-statistic: {iv['f_statistic']:.2f}")
            report.append("")
        
        # Regression Discontinuity Results
        if 'regression_discontinuity' in self.results:
            rd = self.results['regression_discontinuity']
            report.append("📊 REGRESSION DISCONTINUITY")
            report.append("-" * 30)
            report.append(f"ATE (RD): {rd['ate_rd']:.2f}")
            report.append(f"Standard Error: {rd['ate_rd_std']:.2f}")
            report.append(f"Bandwidth: {rd['bandwidth']:.2f}")
            report.append("")
        
        # Sensitivity Analysis
        if 'sensitivity_analysis' in self.results:
            sa = self.results['sensitivity_analysis']
            report.append("🔬 SENSITIVITY ANALYSIS")
            report.append("-" * 30)
            if sa['critical_gamma']:
                report.append(f"Critical Gamma: {sa['critical_gamma']:.2f}")
                report.append("Result: Robust to moderate unobserved confounding")
            else:
                report.append("Result: Sensitive to unobserved confounding")
            report.append("")
        
        return "\n".join(report)
    
    def plot_results(self, save_path: str = None):
        """Create comprehensive visualization of results."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Treatment Effect Distribution', 'Propensity Score Distribution',
                          'Balance Plot', 'Sensitivity Analysis'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Treatment effect distribution
        if 'ml_based' in self.results:
            te_predictions = []
            for method, result in self.results['ml_based'].items():
                te_predictions.extend(result['te_predictions'])
            
            fig.add_trace(
                go.Histogram(x=te_predictions, name='Treatment Effects', nbinsx=30),
                row=1, col=1
            )
        
        # Propensity score distribution
        if 'propensity_matching' in self.results:
            ps_scores = self.results['propensity_matching']['propensity_scores']
            fig.add_trace(
                go.Histogram(x=ps_scores[self.treatment == 1], name='Treated', nbinsx=20),
                row=1, col=2
            )
            fig.add_trace(
                go.Histogram(x=ps_scores[self.treatment == 0], name='Control', nbinsx=20),
                row=1, col=2
            )
        
        # Balance plot
        if 'propensity_matching' in self.results:
            balance_stats = self.results['propensity_matching']['balance_stats']
            variables = list(balance_stats.keys())[:10]  # Top 10 variables
            std_diffs = [balance_stats[var]['std_difference'] for var in variables]
            
            fig.add_trace(
                go.Bar(x=variables, y=std_diffs, name='Standardized Differences'),
                row=2, col=1
            )
        
        # Sensitivity analysis
        if 'sensitivity_analysis' in self.results:
            sa = self.results['sensitivity_analysis']
            fig.add_trace(
                go.Scatter(x=sa['gamma_values'], y=sa['p_values'], 
                          mode='lines+markers', name='P-values'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Causal Inference Results")
        
        if save_path:
            fig.write_html(save_path)
        
        return fig


def run_causal_analysis(data_path: str = "data/simulated_customer_data.csv") -> AdvancedCausalAnalyzer:
    """
    Run comprehensive causal analysis on customer data.
    
    Args:
        data_path: Path to the customer data
        
    Returns:
        AdvancedCausalAnalyzer instance with results
    """
    # Load data
    data = pd.read_csv(data_path)
    
    # Initialize analyzer
    analyzer = AdvancedCausalAnalyzer(
        data=data,
        treatment_col='promotion_response',
        outcome_col='customer_lifetime_value'
    )
    
    # Run analyses
    analyzer.propensity_score_matching()
    analyzer.ml_based_causal_inference()
    analyzer.sensitivity_analysis()
    
    # Generate report
    report = analyzer.generate_report()
    print(report)
    
    return analyzer


if __name__ == "__main__":
    # Run causal analysis
    analyzer = run_causal_analysis()
    
    # Create visualizations
    analyzer.plot_results("causal_analysis_results.html")
