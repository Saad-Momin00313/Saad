import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import shap
import os
from dotenv import load_dotenv
import json
from datetime import datetime
import dask.dataframe as dd
from pathlib import Path
import tempfile
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

# Load environment variables
load_dotenv()

# Configure Gemini AI
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

class DynamicChurnPredictor:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-pro')
        self.data = None
        self.column_types = {}
        self.column_descriptions = {}
        self.numeric_columns = []
        self.categorical_columns = []
        self.date_columns = []
        self.text_columns = []
        self.target_column = None
        self.ml_model = RandomForestClassifier(n_estimators=100, random_state=42)

    def _identify_target_column(self, df):
        """Use AI to identify the most likely target column for churn prediction"""
        # Prepare dataset information
        dataset_info = {
            'columns': {},
            'total_rows': len(df)
        }
        
        for col in df.columns:
            unique_values = df[col].nunique()
            sample_values = df[col].dropna().astype(str).head().tolist()
            
            dataset_info['columns'][col] = {
                'unique_count': int(unique_values),
                'sample_values': sample_values,
                'dtype': str(df[col].dtype),
                'missing_count': int(df[col].isnull().sum()),
                'binary': unique_values == 2
            }

        # Craft prompt for target column identification
        target_prompt = f"""
        Analyze this dataset and identify the most likely target column for churn prediction.
        
        Dataset Information:
        {json.dumps(dataset_info, indent=2)}
        
        Consider these factors:
        1. Churn columns typically have binary/boolean values
        2. Common names include: 'churn', 'churned', 'is_churned', 'status', 'customer_status'
        3. Usually has 2-3 unique values like Yes/No, 1/0, True/False
        4. May indicate customer state/status
        
        Respond in JSON format:
        {{
            "target_column": "column_name",
            "confidence": "high/medium/low",
            "reasoning": "explanation",
            "column_type": "binary/categorical"
        }}
        """

        try:
            response = self.model.generate_content(target_prompt)
            target_analysis = json.loads(response.text)
            
            identified_target = target_analysis.get('target_column')
            if identified_target in df.columns:
                self.target_column = identified_target
                st.info(f"AI identified '{identified_target}' as the target column ({target_analysis.get('confidence')} confidence)")
                st.write(f"Reasoning: {target_analysis.get('reasoning')}")
                return True
                
            st.warning("AI couldn't confidently identify a target column. Please select one manually.")
            # Let user select target column
            self.target_column = st.selectbox(
                "Select target column for churn prediction:",
                options=df.columns.tolist(),
                index=0
            )
            return True
            
        except Exception as e:
            st.error(f"Error identifying target column: {e}")
            # Fallback to user selection
            self.target_column = st.selectbox(
                "Select target column for churn prediction:",
                options=df.columns.tolist(),
                index=0
            )
            return True
    
    def _detect_column_types_with_ai(self, df):
        """Use Gemini AI to intelligently detect column types and generate descriptions"""
        # First identify target column
        if not self._identify_target_column(df):
            return None
            
        # Sample first 100 rows to minimize token usage
        sample_df = df.head(100)
        
        # Prepare column information for AI analysis
        column_info = {}
        for col in df.columns:
            sample_values = sample_df[col].dropna().astype(str).tolist()[:5]
            
            column_info[col] = {
                'sample_values': sample_values,
                'dtype': str(df[col].dtype),
                'unique_count': int(df[col].nunique()),
                'missing_count': int(df[col].isnull().sum()),
                'is_target': col == self.target_column
            }
        
        # Define the prompt with the column information
        prompt_text = f"""
        Analyze the following columns and provide:
        1. Column Type (numeric, categorical, date, text)
        2. Potential role in churn analysis
        3. Brief description of the column
        4. Suggested preprocessing steps
        
        Target Column: {self.target_column}
        
        Column Information:
        {json.dumps(column_info, indent=2)}
        
        Please respond in strict JSON format:
        {{
            "column_name": {{
                "type": "detected_type",
                "churn_potential": "high/medium/low",
                "description": "column description",
                "preprocessing": "suggested steps"
            }}
        }}
        """
        
        try:
            response = self.model.generate_content(prompt_text)
            
            try:
                column_analysis = json.loads(response.text)
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                if json_match:
                    column_analysis = json.loads(json_match.group(0))
                else:
                    st.error("Failed to parse AI response. Please check the input data.")
                    return None
            
            # Reset lists to avoid duplicates
            self.numeric_columns = []
            self.categorical_columns = []
            self.date_columns = []
            self.text_columns = []
            
            # Process AI's column type recommendations
            for col, details in column_analysis.items():
                if col in df.columns:
                    # Special handling for target column
                    if col == self.target_column:
                        self.categorical_columns.append(col)
                        self.column_types[col] = 'categorical'
                        self.column_descriptions[col] = {
                            'type': 'categorical',
                            'churn_potential': 'target',
                            'description': 'Target variable indicating churn status',
                            'preprocessing': 'Convert to binary values (0/1)'
                        }
                        continue
                    
                    # Categorize other columns
                    type_mapping = {
                        'numeric': self.numeric_columns,
                        'categorical': self.categorical_columns,
                        'date': self.date_columns,
                        'text': self.text_columns
                    }
                    
                    column_type = details.get('type', 'text').lower()
                    if column_type in type_mapping:
                        type_mapping[column_type].append(col)
                    
                    self.column_descriptions[col] = {
                        'type': column_type,
                        'churn_potential': details.get('churn_potential', 'low'),
                        'description': details.get('description', 'No description available'),
                        'preprocessing': details.get('preprocessing', 'No specific preprocessing')
                    }
                    
                    self.column_types[col] = column_type
            
            return column_analysis
            
        except Exception as e:
            st.error(f"AI Column Type Detection Failed: {e}")
            self._detect_column_types_fallback(df)
            return None
    
    def _detect_column_types_fallback(self, df):
        """Fallback method for column type detection using pandas dtypes"""
        # Reset lists to avoid duplicates
        self.numeric_columns = []
        self.categorical_columns = []
        self.date_columns = []
        self.text_columns = []
        
        for col in df.columns:
            if col == 'Churn':
                self.categorical_columns.append(col)
                self.column_types[col] = 'categorical'
                continue
                
            if pd.api.types.is_numeric_dtype(df[col]):
                self.numeric_columns.append(col)
                self.column_types[col] = 'numeric'
            elif pd.api.types.is_categorical_dtype(df[col]) or df[col].nunique() < len(df) * 0.1:
                self.categorical_columns.append(col)
                self.column_types[col] = 'categorical'
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                self.date_columns.append(col)
                self.column_types[col] = 'date'
            else:
                self.text_columns.append(col)
                self.column_types[col] = 'text'
    
    def preprocess_data(self, df):
        """Dynamically preprocess data based on AI and fallback column type detection"""
        # Create a copy to avoid modifying original DataFrame
        df_processed = df.copy()
        
        # Special handling for target column (convert Yes/No to 1/0)
        if self.target_column:
            if df_processed[self.target_column].dtype == object:
                # Convert Yes/No to 1/0
                df_processed[self.target_column] = df_processed[self.target_column].map({'Yes': 1, 'No': 0})
        
        # Special handling for TotalCharges column
        if 'TotalCharges' in df_processed.columns:
            # Remove whitespace and convert to numeric
            df_processed['TotalCharges'] = df_processed['TotalCharges'].str.strip()
            df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
        
        # Handle missing and non-numeric values in numeric columns
        for col in self.numeric_columns:
            if col != self.target_column:
                # Convert column to numeric, coercing errors to NaN
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
            
                # Fill NaN values with median
                df_processed[col].fillna(df_processed[col].median(), inplace=True)
        
        # Handle missing values
        for col in df_processed.columns:
            if df_processed[col].isnull().sum() > 0:
                if self.column_types.get(col) == 'numeric':
                    df_processed[col].fillna(df_processed[col].median(), inplace=True)
                elif self.column_types.get(col) == 'categorical':
                    df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
                elif self.column_types.get(col) == 'text':
                    df_processed[col].fillna('Unknown', inplace=True)
        
        # Encode categorical variables
        label_encoders = {}
        for col in self.categorical_columns:
            if col != self.target_column:
                label_encoders[col] = LabelEncoder()
                df_processed[col] = label_encoders[col].fit_transform(df_processed[col].astype(str))
        
        # Separate features and target
        X = df_processed.drop(columns=[self.target_column])
        y = df_processed[self.target_column]
        
        # Standardize numeric features
        numeric_features = [col for col in self.numeric_columns if col != self.target_column]
        if numeric_features:
            scaler = StandardScaler()
            X[numeric_features] = scaler.fit_transform(X[numeric_features])
        
        return X, y, label_encoders
    
    def generate_churn_analysis(self, df):
        """Generate comprehensive churn analysis using Gemini AI"""
        # Preprocess data
        X, y, _ = self.preprocess_data(df)
        
        # Train model
        self.ml_model.fit(X, y)
        model_accuracy = self.ml_model.score(X, y)
        feature_importances = pd.DataFrame({
            'feature': X.columns,
            'importance': self.ml_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Prepare context for AI analysis
        analysis_context = {
            'total_customers': len(df),
            'model_accuracy': model_accuracy,
            'churn_rate': y.mean(),
            'top_features': feature_importances.to_dict('records'),
            'column_descriptions': self.column_descriptions
        }
        
        # Craft detailed prompt for comprehensive analysis
        prompt = f"""
        Perform an in-depth customer churn analysis based on the following insights:

        Dataset Context:
        - Total Customers: {analysis_context['total_customers']:,}
        - Model Accuracy: {analysis_context['model_accuracy']:.2%}
        - Churn Rate: {analysis_context['churn_rate']:.2%}

        Top Predictive Features:
        {json.dumps(analysis_context['top_features'], indent=2)}

        Column Insights:
        {json.dumps(analysis_context['column_descriptions'], indent=2)}

        Provide a comprehensive analysis covering:

        1. CHURN RISK DYNAMICS
        - Primary drivers of customer churn
        - High-risk customer segments
        - Interconnected factors influencing churn

        2. STRATEGIC RECOMMENDATIONS
        - Data-driven retention strategies
        - Targeted intervention approaches
        - Personalization tactics

        3. PREDICTIVE INSIGHTS
        - Future churn probability trends
        - Early warning indicators
        - Risk mitigation strategies

        4. BUSINESS IMPACT ANALYSIS
        - Potential revenue implications
        - Cost of customer acquisition vs retention
        - Long-term customer lifecycle management

        Provide actionable, quantitative insights with specific recommendations.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Analysis generation failed: {e}"
    
    def visualize_insights(self, df):
        """Create comprehensive visualizations"""
        X, y, _ = self.preprocess_data(df)
        
        # Train the model before visualization
        self.ml_model.fit(X, y)
        
        plt.figure(figsize=(15, 10))
        
        # Feature Importance
        plt.subplot(2, 2, 1)
        feature_importances = pd.DataFrame({
            'feature': X.columns,
            'importance': self.ml_model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        sns.barplot(x='importance', y='feature', data=feature_importances)
        plt.title('Top 10 Churn Predictive Features')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        
        return plt

def main():
    st.set_page_config(page_title="Churn Analysis", layout="wide")
    st.title("Customer Churn Analysis")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read CSV with flexible parsing
            df = pd.read_csv(uploaded_file, low_memory=False, parse_dates=True)
            
            # Initialize predictor
            predictor = DynamicChurnPredictor()
            
            # Detect column types
            st.write("### 🕵️ Column Type Detection")
            predictor._detect_column_types_with_ai(df)
            
            # Display column details
            col1, col2 = st.columns(2)
            with col1:
                st.write("#### 📊 Column Types")
                for col, type_ in predictor.column_types.items():
                    st.text(f"{col}: {type_}")
            
            with col2:
                st.write("#### 🎯 Target Column")
                st.write(f"Selected Target: {predictor.target_column}")
            
            # Tabs for different analyses
            tab1, tab2 = st.tabs(["Churn Insights", "Strategic Analysis"])
            
            with tab1:
                st.write("### 🔍 Churn Risk Insights")
                if st.button("Generate Churn Insights"):
                    with st.spinner("Generating comprehensive insights..."):
                        insights = predictor.generate_churn_analysis(df)
                        st.write(insights)
            
            with tab2:
                st.write("### 📈 Strategic Churn Mitigation")
                if st.button("Generate Strategic Report"):
                    with st.spinner("Developing strategic analysis..."):
                        strategy = predictor.generate_churn_analysis(df)
                        st.write(strategy)
            
            
        
        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.error(traceback.format_exc())

if __name__ == "__main__":
    main()