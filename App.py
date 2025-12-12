import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Incident Analysis Dashboard", layout="wide")

st.title("🔧 Incident Analysis Dashboard")
st.markdown("### Comprehensive NLP and Predictive Analysis (2022+ Data)")

# Load data
@st.cache_data
def load_data():
    df = pd.read_excel('Incidents.xlsx')
    
    # Find columns
    date_columns = [col for col in df.columns if 'date' in col.lower()]
    desc_columns = [col for col in df.columns if 'desc' in col.lower() or 'brief' in col.lower()]
    
    incident_date_col = 'Incident Date' if 'Incident Date' in df.columns else date_columns[0]
    brief_desc_col = 'Brief Description' if 'Brief Description' in df.columns else desc_columns[0]
    
    # Convert dates
    df[incident_date_col] = pd.to_datetime(df[incident_date_col], errors='coerce')
    df = df.dropna(subset=[incident_date_col, brief_desc_col])
    
    # Filter: Only 2022 onwards
    df['Year'] = df[incident_date_col].dt.year
    df = df[df['Year'] >= 2022].copy()
    
    df['Month'] = df[incident_date_col].dt.month
    df['Month_Name'] = df[incident_date_col].dt.strftime('%B')
    df['YearMonth'] = df[incident_date_col].dt.to_period('M')
    
    return df, incident_date_col, brief_desc_col

try:
    df, incident_date_col, brief_desc_col = load_data()
    st.success(f"✅ Loaded {len(df)} incidents from {df[incident_date_col].min().date()} to {df[incident_date_col].max().date()} (2022+)")
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

# Extract bigrams
vectorizer = CountVectorizer(ngram_range=(2,2), stop_words='english', max_features=25)
X = vectorizer.fit_transform(df[brief_desc_col].fillna(''))

sum_words = X.sum(axis=0)
bigrams = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
bigrams = sorted(bigrams, key=lambda x: x[1], reverse=True)

bigram_df = pd.DataFrame(bigrams, columns=["bigram", "count"])
top_bigrams = bigram_df['bigram'].tolist()

# Get primary bigram for each incident
def get_primary_bigram(text):
    if pd.isna(text):
        return None
    text = str(text).lower()
    for bigram in top_bigrams:
        if bigram in text:
            return bigram
    return None

df['primary_bigram'] = df[brief_desc_col].apply(get_primary_bigram)

# ==================== FOUR TABS AT THE TOP ====================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Incident Overview", "🗓️ Bigram Frequency by Month", "🔮 2026 Predictions", "⚠️ Severity Mismatch Detector"])

# ==================== TAB 1: INCIDENT OVERVIEW ====================
with tab1:
    st.header("📊 Incident Overview")
    
    col1, col2 = st.columns(2)
    
    # Left: Top 25 Bigrams (HALF SIZE)
    with col1:
        st.subheader("Top 25 Most Common Bigrams")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.barplot(data=bigram_df, x="count", y="bigram", palette="plasma", ax=ax1)
        ax1.set_title("Top 25 Bigrams", fontsize=11, fontweight='bold')
        ax1.set_xlabel("Frequency", fontsize=9)
        ax1.set_ylabel("Bigram", fontsize=9)
        ax1.tick_params(labelsize=7)
        plt.tight_layout()
        st.pyplot(fig1)
    
    # Right: Average Days Open (HALF SIZE)
    with col2:
        st.subheader("Average Days Open by Bigram")
        
        # Find the Date Closed column (might have extra text)
        date_closed_col = None
        for col in df.columns:
            if 'date closed' in col.lower():
                date_closed_col = col
                break
        
        # Calculate days open using Date Closed
        if date_closed_col:
            df['Date_Closed'] = pd.to_datetime(df[date_closed_col], errors='coerce')
            df['days_open'] = (df['Date_Closed'] - df[incident_date_col]).dt.days
            df['days_open'] = df['days_open'].fillna(0).clip(lower=0)
            
            # Only include bigrams that have incidents with valid days_open
            duration_df = df[(df['primary_bigram'].notna()) & (df['days_open'] > 0)].groupby('primary_bigram')['days_open'].mean().reset_index()
            duration_df.columns = ['bigram', 'avg_days_open']
            duration_df = duration_df.sort_values('avg_days_open', ascending=False)
            
            if len(duration_df) > 0:
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                sns.barplot(data=duration_df, x="avg_days_open", y="bigram", palette="magma", ax=ax2)
                ax2.set_title("Avg Days Open", fontsize=11, fontweight='bold')
                ax2.set_xlabel("Average Days Open", fontsize=9)
                ax2.set_ylabel("Bigram", fontsize=9)
                ax2.tick_params(labelsize=7)
                plt.tight_layout()
                st.pyplot(fig2)
            else:
                st.warning("⚠️ No closed incidents with valid dates found.")
        else:
            st.warning("⚠️ 'Date Closed' column not found in the data.")
    
    st.info("ℹ️ **Why some bigrams are missing:** Only bigrams from closed incidents (with Date Closed filled) are shown.")

# ==================== TAB 2: BIGRAM FREQUENCY BY MONTH ====================
with tab2:
    st.header("🗓️ Bigram Frequency by Month")
    
    # Year selector
    years_available = sorted(df['Year'].unique())
    year_options = ['All Years'] + [str(int(year)) for year in years_available]
    
    selected_year = st.selectbox("Select Year:", year_options)
    
    # Filter by year
    if selected_year == 'All Years':
        filtered_df = df.copy()
        title_suffix = "ALL YEARS (2022+)"
    else:
        filtered_df = df[df['Year'] == int(selected_year)].copy()
        title_suffix = f"YEAR {selected_year}"
    
    # Create month matrix
    months_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                    'July', 'August', 'September', 'October', 'November', 'December']
    
    # Create month matrix - FORCE ALL 25 BIGRAMS TO SHOW
    bigram_month_matrix = []
    for bigram in top_bigrams:
        row = []
        for month in months_order:
            month_data = filtered_df[filtered_df['Month_Name'] == month]
            count = sum(month_data[brief_desc_col].fillna('').str.contains(bigram, case=False, regex=False))
            row.append(count)
        bigram_month_matrix.append(row)
    
    result_df = pd.DataFrame(bigram_month_matrix, columns=months_order, index=top_bigrams)
    
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    sns.heatmap(result_df, annot=True, fmt='g', cmap='YlOrRd', 
                linewidths=0.2, cbar_kws={'label': 'Frequency'}, ax=ax3, annot_kws={'fontsize': 5})
    ax3.set_title(f'Bigram Frequency by Month - ALL 25 BIGRAMS ({title_suffix})', 
                  fontsize=10, fontweight='bold', pad=6)
    ax3.set_xlabel('Month', fontsize=8, fontweight='bold')
    ax3.set_ylabel('Bigram', fontsize=8, fontweight='bold')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=6)
    ax3.set_yticklabels(top_bigrams, fontsize=6)
    plt.tight_layout()
    st.pyplot(fig3)
    
    st.info(f"📊 Showing all {len(top_bigrams)} bigrams.")

# ==================== TAB 3: 2026 PREDICTIONS WITH MODEL SELECTION ====================
with tab3:
    st.header("🔮 2026 Incident Predictions")
    
    # Initialize session state for model selection if not exists
    if 'selected_model_index' not in st.session_state:
        st.session_state.selected_model_index = 0
    
    # MODEL SELECTOR - USING SELECTBOX INSTEAD OF RADIO
    st.subheader("Select Prediction Model")
    
    model_options = [
        "📆 Seasonal Average (Recommended)",
        "📊 Weighted Seasonal Average", 
        "📈 Trend-Adjusted Seasonal"
    ]
    
    selected_model = st.selectbox(
        "Choose prediction method:",
        model_options,
        index=st.session_state.selected_model_index,
        key="model_selector_unique_key_tab3"
    )
    
    # Update session state
    st.session_state.selected_model_index = model_options.index(selected_model)
    
    # Clean model name
    if "Seasonal Average" in selected_model and "Weighted" not in selected_model:
        model_name = "Seasonal Average"
    elif "Weighted" in selected_model:
        model_name = "Weighted Seasonal Average"
    else:
        model_name = "Trend-Adjusted Seasonal"
    
    # Prepare monthly counts
    monthly_counts = df.groupby(['YearMonth', 'primary_bigram']).size().reset_index(name='count')
    monthly_counts['YearMonth'] = monthly_counts['YearMonth'].dt.to_timestamp()
    
    # Create predictions
    all_predictions = []
    model_scores = []
    
    for incident_type in top_bigrams:
        type_data = monthly_counts[monthly_counts['primary_bigram'] == incident_type].copy()
        type_data = type_data.sort_values('YearMonth')
        
        if len(type_data) < 6:
            for month in range(1, 13):
                all_predictions.append({
                    'Incident Type': incident_type,
                    'Month': month,
                    'Predicted Count': 0
                })
            continue
        
        type_data['month'] = type_data['YearMonth'].dt.month
        type_data['year'] = type_data['YearMonth'].dt.year
        
        if model_name == "Seasonal Average":
            # Calculate monthly averages
            monthly_avg = type_data.groupby('month')['count'].mean().to_dict()
            
            for month in range(1, 13):
                prediction = monthly_avg.get(month, 0)
                prediction = max(0, int(round(prediction)))
                
                all_predictions.append({
                    'Incident Type': incident_type,
                    'Month': month,
                    'Predicted Count': prediction
                })
            
            # Calculate R² for accuracy
            if len(type_data) > 0:
                predictions = [monthly_avg.get(m, 0) for m in type_data['month']]
                actuals = type_data['count'].values
                ss_res = np.sum((actuals - predictions) ** 2)
                ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                model_scores.append(max(0, r2))
                
        elif model_name == "Weighted Seasonal Average":
            # Weighted average giving more importance to recent years
            max_year = type_data['year'].max()
            
            def calculate_weight(year):
                if year == max_year:
                    return 3
                elif year == max_year - 1:
                    return 2
                else:
                    return 1
            
            type_data['weight'] = type_data['year'].apply(calculate_weight)
            
            # Calculate weighted monthly averages
            monthly_weighted_avg = {}
            for month in range(1, 13):
                month_data = type_data[type_data['month'] == month]
                if len(month_data) > 0:
                    weighted_sum = (month_data['count'] * month_data['weight']).sum()
                    weight_sum = month_data['weight'].sum()
                    monthly_weighted_avg[month] = weighted_sum / weight_sum if weight_sum > 0 else 0
                else:
                    monthly_weighted_avg[month] = 0
            
            for month in range(1, 13):
                prediction = monthly_weighted_avg.get(month, 0)
                prediction = max(0, int(round(prediction)))
                
                all_predictions.append({
                    'Incident Type': incident_type,
                    'Month': month,
                    'Predicted Count': prediction
                })
            
            # Calculate R² for accuracy
            if len(type_data) > 0:
                predictions = [monthly_weighted_avg.get(m, 0) for m in type_data['month']]
                actuals = type_data['count'].values
                ss_res = np.sum((actuals - predictions) ** 2)
                ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                model_scores.append(max(0, r2))
                
        else:  # Trend-Adjusted Seasonal
            # Calculate seasonal averages
            monthly_avg = type_data.groupby('month')['count'].mean().to_dict()
            
            # Calculate yearly trend (simple linear trend)
            years = type_data['year'].unique()
            if len(years) >= 2:
                yearly_totals = type_data.groupby('year')['count'].sum()
                year_values = yearly_totals.index.values
                total_values = yearly_totals.values
                
                # Simple linear regression for trend
                n = len(year_values)
                sum_x = np.sum(year_values)
                sum_y = np.sum(total_values)
                sum_xy = np.sum(year_values * total_values)
                sum_x2 = np.sum(year_values ** 2)
                
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2) if (n * sum_x2 - sum_x ** 2) != 0 else 0
                intercept = (sum_y - slope * sum_x) / n
                
                # Calculate trend factor for 2026
                last_year = type_data['year'].max()
                last_year_total = type_data[type_data['year'] == last_year]['count'].sum()
                predicted_2026_total = slope * 2026 + intercept
                
                # Apply trend factor to seasonal averages
                trend_factor = predicted_2026_total / last_year_total if last_year_total > 0 else 1
                trend_factor = max(0.5, min(2.0, trend_factor))
            else:
                trend_factor = 1.0
            
            for month in range(1, 13):
                base_prediction = monthly_avg.get(month, 0)
                prediction = base_prediction * trend_factor
                prediction = max(0, int(round(prediction)))
                
                all_predictions.append({
                    'Incident Type': incident_type,
                    'Month': month,
                    'Predicted Count': prediction
                })
            
            # Calculate R² for accuracy
            if len(type_data) > 0:
                predictions = [monthly_avg.get(m, 0) for m in type_data['month']]
                actuals = type_data['count'].values
                ss_res = np.sum((actuals - predictions) ** 2)
                ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                model_scores.append(max(0, r2))
    
    avg_accuracy = np.mean(model_scores) if model_scores else 0
    
    # Model info
    model_descriptions = {
        "Seasonal Average": "📆 Simple average for each month across all years. Most stable and reliable.",
        "Weighted Seasonal Average": "📊 Recent years weighted more (3x current, 2x previous, 1x older). Adapts to recent changes.",
        "Trend-Adjusted Seasonal": "📈 Seasonal average adjusted by overall yearly trend. Accounts for growth/decline patterns."
    }
    
    st.markdown(f"""
    ### 📋 Model: {model_name}
    **Description:** {model_descriptions[model_name]}
    **Accuracy (R²):** {avg_accuracy:.1%}
    """)
    
    st.markdown("---")
    
    predictions_df = pd.DataFrame(all_predictions)
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Force all 25 bigrams
    complete_data = []
    for bigram in top_bigrams:
        for month_num in range(1, 13):
            matching = predictions_df[
                (predictions_df['Incident Type'] == bigram) & 
                (predictions_df['Month'] == month_num)
            ]
            
            count = matching['Predicted Count'].iloc[0] if len(matching) > 0 else 0
            
            complete_data.append({
                'Incident Type': bigram,
                'Month': month_num,
                'Predicted Count': count
            })
    
    complete_df = pd.DataFrame(complete_data)
    heatmap_data = complete_df.pivot(index='Incident Type', columns='Month', values='Predicted Count')
    heatmap_data.columns = month_names
    
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    sns.heatmap(heatmap_data, annot=True, fmt='g', cmap='Reds', 
                linewidths=0.2, cbar_kws={'label': 'Count'}, ax=ax4, annot_kws={'fontsize': 5})
    ax4.set_title(f'2026 Predictions - ALL 25 BIGRAMS ({model_name})', 
                  fontsize=10, fontweight='bold', pad=6)
    ax4.set_xlabel('Month', fontsize=8, fontweight='bold')
    ax4.set_ylabel('Incident Type', fontsize=8, fontweight='bold')
    ax4.set_xticklabels(month_names, rotation=45, ha='right', fontsize=6)
    ax4.set_yticklabels(heatmap_data.index, fontsize=6)
    plt.tight_layout()
    st.pyplot(fig4)
    
    # Summary
    st.subheader("📋 Top 10 Predictions")
    summary_data = []
    for incident_type in top_bigrams:
        type_preds = predictions_df[predictions_df['Incident Type'] == incident_type]
        total = type_preds['Predicted Count'].sum()
        if total > 0:
            peak_row = type_preds.loc[type_preds['Predicted Count'].idxmax()]
            summary_data.append({
                'Incident Type': incident_type,
                'Total 2026': int(total),
                'Peak Month': month_names[int(peak_row['Month']) - 1],
                'Peak Count': int(peak_row['Predicted Count'])
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data).sort_values('Total 2026', ascending=False).head(10)
        st.dataframe(summary_df, use_container_width=True)
    
    csv = predictions_df.to_csv(index=False)
    st.download_button(
        label=f"📥 Download {model_name} Predictions",
        data=csv,
        file_name=f"predictions_2026_{model_name.replace(' ', '_')}.csv",
        mime="text/csv"
    )

# ==================== TAB 4: SEVERITY MISMATCH DETECTOR ====================
with tab4:
    st.header("⚠️ Severity Mismatch Detector")
    st.markdown("### Find incidents where text severity doesn't match risk ranking")
    
    # Define severity keywords
    HIGH_SEVERITY_WORDS = [
        'major', 'critical', 'severe', 'serious', 'catastrophic', 'emergency',
        'explosion', 'fire', 'fatality', 'death', 'injury', 'hospitalized',
        'massive', 'extensive', 'significant', 'large', 'total', 'complete',
        'failure', 'rupture', 'burst', 'collapse'
    ]
    
    LOW_SEVERITY_WORDS = [
        'minor', 'small', 'slight', 'negligible', 'minimal', 'insignificant',
        'cosmetic', 'superficial', 'light', 'low'
    ]
    
    # Function to calculate text severity score
    def calculate_text_severity(text):
        if pd.isna(text):
            return 0
        text_lower = str(text).lower()
        
        high_count = sum(1 for word in HIGH_SEVERITY_WORDS if word in text_lower)
        low_count = sum(1 for word in LOW_SEVERITY_WORDS if word in text_lower)
        
        # Net severity score
        score = high_count - low_count
        return score
    
    # Function to convert risk ranking to numeric score
    def risk_to_numeric(risk_str):
        if pd.isna(risk_str):
            return 0
        risk_str = str(risk_str).upper()
        
        # Extract the number from risk ranking (e.g., "08 - Green" -> 8)
        try:
            number = int(''.join(filter(str.isdigit, risk_str)))
        except:
            number = 0
        
        # Categorize: Green (low), Orange (medium), Red (high)
        if 'RED' in risk_str or number >= 16:
            return 3  # High risk
        elif 'ORANGE' in risk_str or number >= 10:
            return 2  # Medium risk
        else:
            return 1  # Low risk (Green)
    
    # Calculate scores for all incidents
    df['text_severity_score'] = df[brief_desc_col].apply(calculate_text_severity)
    df['risk_numeric'] = df['Highest Initial Risk Ranking'].apply(risk_to_numeric)
    
    # Identify mismatches
    # HIGH TEXT SEVERITY but LOW RISK = Potentially under-reported
    df['under_reported'] = (df['text_severity_score'] >= 2) & (df['risk_numeric'] == 1)
    
    # LOW TEXT SEVERITY but HIGH RISK = Potentially over-reported (less critical)
    df['over_reported'] = (df['text_severity_score'] <= -1) & (df['risk_numeric'] >= 2)
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        under_count = df['under_reported'].sum()
        st.metric("🚨 Potentially Under-Reported", under_count, 
                  help="High severity words in description but ranked as low risk (Green)")
    
    with col2:
        over_count = df['over_reported'].sum()
        st.metric("📉 Potentially Over-Reported", over_count,
                  help="Low severity words in description but ranked as medium/high risk")
    
    with col3:
        aligned = len(df) - under_count - over_count
        aligned_pct = (aligned / len(df)) * 100
        st.metric("✅ Well-Aligned", f"{aligned_pct:.1f}%",
                  help="Incidents where text severity matches risk ranking")
    
    st.markdown("---")
    
    # Show under-reported incidents
    st.subheader("🚨 Under-Reported Incidents (Action Needed)")
    st.markdown("*These incidents have severe language in descriptions but are classified as LOW RISK (Green)*")
    
    under_reported = df[df['under_reported']].copy()
    
    if len(under_reported) > 0:
        # Add severity indicator column
        def get_severity_words_found(text):
            if pd.isna(text):
                return ""
            text_lower = str(text).lower()
            found = [word for word in HIGH_SEVERITY_WORDS if word in text_lower]
            return ", ".join(found[:3])  # Show top 3 words
        
        under_reported['Severity Words Found'] = under_reported[brief_desc_col].apply(get_severity_words_found)
        
        display_cols = ['Incident No.', 'Incident Date', brief_desc_col, 
                       'Highest Initial Risk Ranking', 'Severity Words Found', 'Status']
        
        # Only show columns that exist
        display_cols = [col for col in display_cols if col in under_reported.columns]
        
        under_reported_display = under_reported[display_cols].sort_values('Incident Date', ascending=False)
        
        st.dataframe(
            under_reported_display.head(20),
            use_container_width=True,
            hide_index=True
        )
        
        if len(under_reported) > 20:
            st.info(f"ℹ️ Showing top 20 of {len(under_reported)} under-reported incidents")
        
        # Download option
        csv = under_reported[display_cols].to_csv(index=False)
        st.download_button(
            label="📥 Download All Under-Reported Incidents",
            data=csv,
            file_name="under_reported_incidents.csv",
            mime="text/csv"
        )
    else:
        st.success("✅ No under-reported incidents found!")
    
    st.markdown("---")
    
    # Show over-reported incidents (collapsible)
    with st.expander("📉 View Over-Reported Incidents (Optional Review)"):
        st.markdown("*These incidents have mild language but are classified as MEDIUM/HIGH RISK*")
        
        over_reported = df[df['over_reported']].copy()
        
        if len(over_reported) > 0:
            over_reported['Low Severity Words Found'] = over_reported[brief_desc_col].apply(
                lambda x: ", ".join([w for w in LOW_SEVERITY_WORDS if w in str(x).lower()][:3])
            )
            
            display_cols_over = ['Incident No.', 'Incident Date', brief_desc_col,
                                'Highest Initial Risk Ranking', 'Low Severity Words Found', 'Status']
            display_cols_over = [col for col in display_cols_over if col in over_reported.columns]
            
            over_reported_display = over_reported[display_cols_over].sort_values('Incident Date', ascending=False)
            
            st.dataframe(
                over_reported_display.head(20),
                use_container_width=True,
                hide_index=True
            )
            
            if len(over_reported) > 20:
                st.info(f"ℹ️ Showing top 20 of {len(over_reported)} over-reported incidents")
        else:
            st.success("✅ No over-reported incidents found!")
    
    st.markdown("---")
    
    # Visualization: Severity vs Risk Distribution
    st.subheader("📊 Text Severity vs Risk Ranking Distribution")
    
    # Create severity categories
    def categorize_text_severity(score):
        if score >= 2:
            return "High Severity Text"
        elif score <= -1:
            return "Low Severity Text"
        else:
            return "Neutral Text"
    
    df['text_severity_category'] = df['text_severity_score'].apply(categorize_text_severity)
    
    # Create risk categories
    def categorize_risk(numeric):
        if numeric == 3:
            return "High Risk (Red)"
        elif numeric == 2:
            return "Medium Risk (Orange)"
        else:
            return "Low Risk (Green)"
    
    df['risk_category'] = df['risk_numeric'].apply(categorize_risk)
    
    # Create crosstab
    cross_tab = pd.crosstab(df['text_severity_category'], df['risk_category'])
    
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    cross_tab.plot(kind='bar', ax=ax5, color=['#90EE90', '#FFA500', '#FF6B6B'])
    ax5.set_title('Text Severity vs Risk Ranking', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Text Severity Category', fontsize=12)
    ax5.set_ylabel('Number of Incidents', fontsize=12)
    ax5.legend(title='Risk Category', loc='upper right')
    ax5.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig5)
    
    st.info("""
    **How to interpret:**
    - 🚨 **Under-reported**: High severity text but Low Risk (Green) - These need review!
    - 📉 **Over-reported**: Low severity text but Medium/High Risk - May be overly cautious
    - ✅ **Well-aligned**: Text severity matches risk ranking
    """)

st.markdown("---")
st.success("✅ Analysis Complete!")