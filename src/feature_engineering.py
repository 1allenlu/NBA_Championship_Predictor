import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureEngineering:
    """Advanced feature engineering for NBA championship prediction"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def create_star_power_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create weighted star power index"""
        weights = {
            'mvp_votes': 0.4,
            'all_nba_first': 0.3,
            'all_nba_second': 0.2,
            'all_star_count': 0.1
        }
        
        df['star_power_index'] = sum(
            df[col] * weight for col, weight in weights.items()
        )
        return df
    
    def create_four_factors_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basketball analytics four factors"""
        # Offensive Four Factors
        df['off_four_factors'] = (
            0.4 * df['efg_pct'] +
            0.25 * df['tov_pct'] +
            0.2 * df['orb_pct'] +
            0.15 * df['ft_rate']
        )
        
        # Defensive Four Factors  
        df['def_four_factors'] = (
            0.4 * (1 - df['opp_efg_pct']) +
            0.25 * df['opp_tov_pct'] +
            0.2 * (1 - df['opp_orb_pct']) +
            0.15 * (1 - df['opp_ft_rate'])
        )
        
        df['net_four_factors'] = df['off_four_factors'] - df['def_four_factors']
        return df
    
    def create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Time-based momentum features"""
        # Last 20 games performance
        df['last_20_win_pct'] = df['last_20_wins'] / 20
        df['last_20_net_rating'] = df['last_20_off_rating'] - df['last_20_def_rating']
        
        # Post-trade deadline performance
        df['post_trade_momentum'] = (
            df['post_trade_win_pct'] - df['pre_trade_win_pct']
        )
        
        return df
    
    def create_playoff_readiness_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Composite playoff readiness metric"""
        components = {
            'playoff_exp_avg': 0.3,
            'clutch_win_pct': 0.25,
            'vs_playoff_teams_pct': 0.25,
            'injury_health_score': 0.2
        }
        
        df['playoff_readiness'] = sum(
            df[col] * weight for col, weight in components.items()
        )
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main feature engineering pipeline"""
        df = self.create_star_power_index(df)
        df = self.create_four_factors_score(df)
        df = self.create_momentum_features(df)
        df = self.create_playoff_readiness_score(df)
        
        # Team chemistry features
        df['payroll_efficiency'] = df['wins'] / (df['payroll_millions'] + 1)
        df['age_balance'] = 1 / (1 + abs(df['avg_age'] - 27))  # Peak age around 27
        
        return df