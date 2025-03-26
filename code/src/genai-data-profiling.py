from io import StringIO
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import google.generativeai as genai
from crewai import Agent, Task, Crew, Process,LLM
from typing import List, Dict, Any
import json


# Initialize Gemini model
model = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.7,
    api_key="AIzaSyCJ3y0JCD0NaZ8zmvqgC7SrC3vFEsjdwgU"
)

class RegulationProcessor:
    """Process regulatory documents and extract validation rules"""
    
    def extract_rules_from_regulations(self, regulation_text: str) -> List[Dict]:
        """Extract validation rules from regulatory text using Gemini"""
        prompt = f"""
        As a financial regulatory expert, analyze the following regulatory reporting instructions:
        
        {regulation_text}
        
        Extract specific data validation rules that must be applied to ensure compliance.
        For each rule, provide:
        1. A unique identifier (rule_id)
        2. A description of the rule in plain English
        3. The data fields this rule applies to
        4. The validation logic expressed as a logical condition
        5. The severity of violation (High, Medium, Low)
        
        Format your response as a JSON array of rule objects.
        """
        
        response = model.generate_content(prompt)
        
        # Parse the response to extract JSON data
        response_text = response.text
        try:
            # Find JSON content within response if needed
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            if json_start >= 0 and json_end > 0:
                json_content = response_text[json_start:json_end]
                extracted_rules = json.loads(json_content)
            else:
                # If not in JSON format, prompt the model again specifically for JSON
                clarification_prompt = "Please reformat your response as a valid JSON array of rule objects."
                clarification = model.generate_content(clarification_prompt)
                json_start = clarification.text.find('[')
                json_end = clarification.text.rfind(']') + 1
                json_content = clarification.text[json_start:json_end]
                extracted_rules = json.loads(json_content)
                
            return extracted_rules
        except json.JSONDecodeError:
            print("Failed to parse JSON response. Using simplified format.")
            # Return a simplified format if JSON parsing fails
            return [{"rule_id": "R001", "description": "Error parsing rules", "fields": ["unknown"], "logic": "N/A", "severity": "High"}]
    
    def generate_validation_code(self, rules: List[Dict]) -> str:
        """Generate Python validation code from extracted rules"""
        prompt = f"""
        Generate Python code that implements the following validation rules:
        
        {json.dumps(rules, indent=2)}
        
        The code should:
        1. Accept a pandas DataFrame as input
        2. Apply each validation rule and identify violations
        3. Return a DataFrame with results of each validation check
        4. Include appropriate comments explaining the validation logic
        
        Make sure the code is optimized, handles edge cases, and includes error handling.
        """
        
        response = model.generate_content(prompt)
        return response.text

class DataProfiler:
    """Profile data using unsupervised learning techniques"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def profile_numeric_columns(self, df: pd.DataFrame, numeric_cols: List[str]) -> Dict:
        """Generate statistical profiles for numeric columns"""
        if not numeric_cols:
            return {}
            
        profiles = {}
        for col in numeric_cols:
            if col in df.columns:
                column_data = df[col].dropna()
                profiles[col] = {
                    "mean": column_data.mean(),
                    "median": column_data.median(),
                    "std": column_data.std(),
                    "min": column_data.min(),
                    "max": column_data.max(),
                    "q1": column_data.quantile(0.25),
                    "q3": column_data.quantile(0.75),
                    "iqr": column_data.quantile(0.75) - column_data.quantile(0.25)
                }
        return profiles
    
    def detect_anomalies(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """Detect anomalies using Isolation Forest"""
        if not numeric_cols or len(numeric_cols) == 0:
            return pd.DataFrame({"error": ["No numeric columns provided"]})
            
        # Filter to only include columns that exist in the dataframe
        valid_cols = [col for col in numeric_cols if col in df.columns]
        
        if len(valid_cols) == 0:
            return pd.DataFrame({"error": ["None of the specified numeric columns exist in the dataframe"]})
            
        # Prepare data
        df_numeric = df[valid_cols].copy()
        df_numeric = df_numeric.fillna(df_numeric.mean())
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(df_numeric)
        
        # Apply Isolation Forest
        isolation_forest = IsolationForest(contamination=0.05, random_state=42)
        df['anomaly_score'] = isolation_forest.fit_predict(scaled_data)
        df['is_anomaly'] = df['anomaly_score'] == -1
        
        return df
    
    def cluster_data(self, df: pd.DataFrame, numeric_cols: List[str], n_clusters=3) -> pd.DataFrame:
        """Cluster data using K-means"""
        # Prepare data
        df_numeric = df[numeric_cols].copy()
        df_numeric = df_numeric.fillna(df_numeric.mean())
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(df_numeric)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(scaled_data)
        
        return df

class RiskScorer:
    """Calculate risk scores for data records based on rule violations"""
    
    def calculate_risk_scores(self, df: pd.DataFrame, validation_results: pd.DataFrame, rules: List[Dict]) -> pd.DataFrame:
        """Calculate risk scores based on rule violations and anomaly detection"""
        # Create severity weights
        severity_weights = {"High": 10, "Medium": 5, "Low": 2}
        
        # Initialize risk score column
        df['risk_score'] = 0
        
        # Add points for each rule violation
        for rule in rules:
            rule_id = rule['rule_id']
            severity = rule['severity']
            weight = severity_weights.get(severity, 1)
            
            if rule_id in validation_results.columns:
                # Add weighted points for each violation
                df['risk_score'] += weight * (~validation_results[rule_id]).astype(int)
        
        # Add additional points for anomalies
        if 'is_anomaly' in df.columns:
            df['risk_score'] += 15 * df['is_anomaly'].astype(int)
        
        # Normalize risk score to 0-100 scale
        max_score = df['risk_score'].max()
        if max_score > 0:
            df['risk_score'] = (df['risk_score'] / max_score) * 100
        
        # Add risk category
        df['risk_category'] = pd.cut(
            df['risk_score'], 
            bins=[0, 20, 40, 70, 100], 
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        return df
    
    def suggest_remediation(self, df: pd.DataFrame, validation_results: pd.DataFrame, rules: List[Dict]) -> pd.DataFrame:
        """Generate remediation suggestions for high-risk records"""
        # Create a copy of the dataframe
        df_remediation = df.copy()
        
        # Initialize remediation column
        df_remediation['remediation_actions'] = ""
        
        # Generate remediation suggestions for each record
        for idx, row in df_remediation.iterrows():
            if row['risk_category'] in ['High', 'Critical']:
                # Collect failed rules
                failed_rules = []
                for rule in rules:
                    rule_id = rule['rule_id']
                    if rule_id in validation_results.columns and not validation_results.loc[idx, rule_id]:
                        failed_rules.append(rule)
                
                # Generate remediation suggestions using Gemini
                if failed_rules:
                    prompt = f"""
                    As a financial compliance expert, suggest specific remediation actions for a transaction with the following rule violations:
                    
                    {json.dumps(failed_rules, indent=2)}
                    
                    Transaction data:
                    {json.dumps(row.to_dict(), indent=2)}
                    
                    Provide a concise bullet-point list of specific remediation actions.
                    """
                    
                    try:
                        response = model.generate_content(prompt)
                        df_remediation.at[idx, 'remediation_actions'] = response.text.strip()
                    except Exception as e:
                        df_remediation.at[idx, 'remediation_actions'] = f"Error generating remediation: {str(e)}"
        
        return df_remediation

# CrewAI Agents
class RegulatoryExpertAgent(Agent):
    def __init__(self):
        super().__init__(
            role="Regulatory Expert",
            goal="Extract validation rules from regulatory instructions",
            backstory="I am an expert in financial regulations with deep knowledge of regulatory reporting requirements.",
            verbose=True,
            llm=model
        )
        self._regulation_processor = RegulationProcessor()
    
    def run(self, task):
        if "extract validation rules" in task.description.lower():
            if task.context and "regulation_text" in task.context:
                return self.regulation_processor.extract_rules_from_regulations(task.context["regulation_text"])
            else:
                return "Error: No regulatory text provided in context"
        elif "generate validation code" in task.description.lower():
            if task.context and "rules" in task.context:
                return self.regulation_processor.generate_validation_code(task.context["rules"])
            else:
                return "Error: No rules provided in context"
        else:
            return f"I can help with regulatory tasks. Please ask me to extract validation rules from regulations or generate validation code from rules."

class DataScientistAgent(Agent):
    def __init__(self):
        super().__init__(
            role="Data Scientist",
            goal="Apply unsupervised learning techniques to detect anomalies and patterns",
            backstory="I specialize in using machine learning to identify patterns and anomalies in financial data.",
            verbose=True,
            llm=model
        )
        self._data_profiler = DataProfiler()
    
    def run(self, task):
        if "apply unsupervised learning" in task.description.lower() or "profile data" in task.description.lower():
            if task.context and "dataframe" in task.context and "numeric_cols" in task.context:
                df = task.context["dataframe"]
                numeric_cols = task.context["numeric_cols"]
                
                # Profile data
                profiles = self.data_profiler.profile_numeric_columns(df, numeric_cols)
                
                # Detect anomalies
                df_anomalies = self.data_profiler.detect_anomalies(df, numeric_cols)
                
                # Cluster data
                df_clusters = self.data_profiler.cluster_data(df_anomalies, numeric_cols)
                
                return {
                    "profiled_dataframe": df_clusters,
                    "data_profiles": profiles
                }
            else:
                return "Error: Missing dataframe or numeric columns in context"
        else:
            return f"I can help with data science tasks. Please ask me to profile data or apply unsupervised learning techniques."

class ComplianceAuditorAgent(Agent):
    def __init__(self):
        super().__init__(
            role="Compliance Auditor",
            goal="Evaluate risk and suggest remediation actions",
            backstory="I have years of experience auditing financial institutions for regulatory compliance.",
            verbose=True,
            llm=model
        )
        self._risk_scorer = RiskScorer()
    
    def run(self, task):
        if "evaluate risk" in task.description.lower() or "suggest remediation" in task.description.lower():
            if task.context and "dataframe" in task.context and "validation_results" in task.context and "rules" in task.context:
                df = task.context["dataframe"]
                validation_results = task.context["validation_results"]
                rules = task.context["rules"]
                
                # Calculate risk scores
                df_with_risk = self._risk_scorer.calculate_risk_scores(df, validation_results, rules)
                
                # Generate remediation suggestions
                df_with_remediation = self._risk_scorer.suggest_remediation(df_with_risk, validation_results, rules)
                
                # Return the actual DataFrame instead of code
                return {
                    "risk_assessment": df_with_remediation,
                    "summary": {
                        "total_risks": len(df_with_remediation),
                        "high_risks": len(df_with_remediation[df_with_remediation['risk_score'] > 7]),
                        "medium_risks": len(df_with_remediation[(df_with_remediation['risk_score'] > 3) & (df_with_remediation['risk_score'] <= 7)]),
                        "low_risks": len(df_with_remediation[df_with_remediation['risk_score'] <= 3])
                    }
                }
            else:
                return "Error: Missing required context for risk evaluation"
        else:
            return f"I can help with compliance and risk assessment tasks. Please ask me to evaluate risk or suggest remediation actions."
class ValidatorAgent(Agent):
    def __init__(self):
        super().__init__(
            role="Data Validator",
            goal="Apply validation rules to data and identify violations",
            backstory="I am an expert in data validation with experience in financial compliance validation.",
            verbose=True,
            llm=model
        )
    
    def run(self, task):
        if "execute validation code" in task.description.lower() or "validate data" in task.description.lower():
            if task.context and "dataframe" in task.context and "rules" in task.context:
                df = task.context["dataframe"]
                rules = task.context["rules"]
                validation_code = task.context.get("validation_code", "")
                
                # Create validation results
                validation_results = pd.DataFrame(index=df.index)
                
                # In real implementation, we would execute the validation code
                # Here we're creating mock results
                for rule in rules:
                    rule_id = rule.get('rule_id', f"R{len(validation_results.columns) + 1}")
                    # Mock validation - in real implementation, would run the generated code
                    validation_results[rule_id] = np.random.choice(
                        [True, False], 
                        size=len(df), 
                        p=[0.9, 0.1]  # 90% pass, 10% fail
                    )
                
                return {
                    "validation_results": validation_results,
                    "violation_count": (~validation_results).sum().sum(),
                    "passing_rate": validation_results.mean().mean() * 100
                }
            else:
                return "Error: Missing dataframe or rules in context"
        else:
            return f"I can help with data validation tasks. Please ask me to validate data against rules."

def create_data_profiling_crew(regulation_text: str, data_file_path: str):
    """Create and execute a CrewAI crew for data profiling"""
    
    # Load the data
    df = pd.read_csv(data_file_path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Initialize agents
    regulatory_expert = RegulatoryExpertAgent()
    data_scientist = DataScientistAgent()
    validator = ValidatorAgent()
    compliance_auditor = ComplianceAuditorAgent()
    
   # Task 1: Extract rules from regulations
    extract_rules_task = Task(
        description="Extract validation rules from the regulatory instructions",
        expected_output="List of validation rules in JSON format",
        context=[{
            "description": "Regulatory text to analyze",
            "expected_output": "Validation rules in JSON format",
            "regulation_text": regulation_text
        }],
        agent=regulatory_expert
    )
    
    # Task 2: Generate validation code
    generate_code_task = Task(
        description="Generate validation code based on the extracted rules",
        expected_output="Python code as string",
        context=[{
            "description": "Generate validation code from rules",
            "expected_output": "Python validation code",
            "rules": lambda: crew.get_task_output(extract_rules_task)
        }],
        agent=regulatory_expert
    )
    
    # Task 3: Profile data using unsupervised learning
    profile_data_task = Task(
        description="Apply unsupervised learning techniques to detect anomalies and patterns in the data",
        expected_output="Dictionary containing profiled dataframe and data profiles",
        context=[{
            "description": "Profile data using ML techniques",
            "expected_output": "Profiled data with anomalies",
            "dataframe": df,
            "numeric_cols": numeric_cols
        }],
        agent=data_scientist
    )
    
    # Task 4: Execute validation code
    execute_validation_task = Task(
        description="Execute validation code against the data",
        expected_output="Dictionary containing validation results and statistics",
        context=[{
            "description": "Execute validation rules",
            "expected_output": "Validation results",
            "dataframe": df,
            "rules": lambda: crew.get_task_output(extract_rules_task),
            "validation_code": lambda: crew.get_task_output(generate_code_task)
        }],
        agent=validator
    )
    
    # Task 5: Evaluate risk and suggest remediation
    evaluate_risk_task = Task(
        description="Evaluate risk and suggest remediation actions based on validation results",
        expected_output="DataFrame with risk scores and remediation suggestions",
        context=[{
            "description": "Evaluate risks and suggest fixes",
            "expected_output": "Risk assessment and remediation suggestions",
            "dataframe": lambda: crew.get_task_output(profile_data_task)["profiled_dataframe"],
            "validation_results": lambda: crew.get_task_output(execute_validation_task)["validation_results"],
            "rules": lambda: crew.get_task_output(extract_rules_task)
        }],
        agent=compliance_auditor
    )
    
    # Create the crew
    crew = Crew(
        agents=[regulatory_expert, data_scientist, validator, compliance_auditor],
        tasks=[extract_rules_task, generate_code_task, profile_data_task, execute_validation_task, evaluate_risk_task],
        verbose=True,
        process=Process.sequential
    )
    
    # Execute the ComplianceAuditorAgent
    result = crew.kickoff()
    return result
def format_validation_results(validation_data):
    """Convert validation results list to formatted table and statistics"""
    # Extract validation results
    validation_results = validation_data.get('validation_results', [])
    statistics = validation_data.get('statistics', {})
    
    # Create DataFrame for validation results
    df = pd.DataFrame(validation_results)
    
    # Format statistics section
    stats_str = "Statistics Summary:\n" + "="*50 + "\n\n"
    stats_str += f"Total Records: {statistics.get('total_records', 0)}\n"
    stats_str += f"Passed Records: {statistics.get('passed_records', 0)}\n"
    stats_str += f"Failed Records: {statistics.get('failed_records', 0)}\n"
    stats_str += f"Pass Rate: {statistics.get('pass_rate', '0%')}\n"
    stats_str += f"Fail Rate: {statistics.get('fail_rate', '0%')}\n"
    stats_str += "\nFields Validated:\n"
    for field in statistics.get('fields_validated', []):
        stats_str += f"- {field}\n"
    stats_str += "\nValidation Rules Applied:\n"
    for rule in statistics.get('validation_rules_applied', []):
        stats_str += f"- {rule}\n"
    stats_str += "\n" + "="*50 + "\n\n"

    return stats_str
def generate_final_report(crew_outputs):
    """Generate a formatted report for tasks 3 (JSON) and 4 (raw string) only"""
    report = """=== Data Validation Report ===\n\n"""
    
    try:
        # Task 3: Validation Results (JSON)
        validation_data = json.loads(crew_outputs.tasks_output[3].raw
                                   .replace('```json', '').replace('```', ''))
        
        report += "Validation Results Summary:\n" + "="*20 + "\n"
        report += "" +format_validation_results(validation_data)
        
       
        
    except json.JSONDecodeError as je:
        report += f"\nJSON Parsing Error: {str(je)}\n"
    except Exception as e:
        report += f"\nError Processing Data: {str(e)}\n"
    
    return report
def main():
    # Sample regulation text
    sample_regulation = """
    Financial Reporting Regulation 2023/45
    
    Section 5.2 - Data Quality Requirements
    
    5.2.1 All reported financial transactions must include a valid transaction ID following the pattern TXN-YYYY-NNNNN where YYYY is the year and NNNNN is a sequence number.
    
    5.2.2 The reported transaction amount must not exceed the authorized limit of $1,000,000 per transaction unless specifically approved by senior management.
    
    5.2.3 Transactions marked as "high-priority" must be processed within 24 hours of receipt.
    
    5.2.4 The sum of debits and credits for each reporting period must balance to zero with a tolerance of 0.01%.
    
    5.2.5 Customer identification fields must not be null for any transaction exceeding $10,000.
    """
    
    # Create sample data
    sample_data = pd.DataFrame({
        'transaction_id': ['TXN-2023-00001', 'TXN-2023-00002', 'TXN-2023-00003', 'TX-2023-00004', 'TXN-2023-00005'],
        'amount': [50000, 1200000, 75000, 9000, 11000],
        'priority': ['normal', 'high', 'normal', 'high', 'normal'],
        'processing_time_hours': [36, 18, 48, 22, 12],
        'customer_id': ['C12345', 'C23456', 'C34567', None, None],
        'debit': [50000, 1200000, 0, 9000, 0],
        'credit': [0, 0, 75000, 0, 11000]
    })
    
    sample_data_path = 'sample_transactions.csv'
    sample_data.to_csv(sample_data_path, index=False)
    
    # Run the crew
    print("Running data profiling using CrewAI approach...")
    results = create_data_profiling_crew(sample_regulation, sample_data_path)
    
      # Generate and display the final report
    report = generate_final_report(results)
    print(report)
    # Optionally save the report to a file
    with open('data_profiling_report.txt', 'w') as f:
        f.write(report)


    # Clean up
    if os.path.exists(sample_data_path):
        os.remove(sample_data_path)

if __name__ == "__main__":
    main()
