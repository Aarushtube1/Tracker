import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import calendar
import io
import base64
from PIL import Image
import os
import json

# Set page configuration
st.set_page_config(
    page_title="Personal Finance Tracker",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define categories and subcategories
CATEGORIES = {
    "Food": ["Groceries", "Dining Out", "Food Delivery", "Snacks"],
    "Transport": ["Public Transport", "Fuel", "Cab/Auto", "Car Maintenance", "Bike Maintenance"],
    "Utilities": ["Electricity", "Water", "Gas", "Internet", "Mobile Recharge"],
    "Education": ["Books", "Courses", "Tuition", "Stationary"],
    "Entertainment": ["Movies", "Music", "Subscriptions", "Games"],
    "Shopping": ["Clothing", "Electronics", "Home Goods", "Personal Care"],
    "Health": ["Medicines", "Doctor Visits", "Fitness", "Insurance"],
    "Housing": ["Rent", "Maintenance", "Furniture", "Home Appliances"],
    "Miscellaneous": ["Gifts", "Donations", "Others"]
}

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #424242;
    }
    .card {
        padding: 20px;
        border-radius: 10px;
        background-color: #f9f9f9;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1976D2;
    }
    .metric-label {
        font-size: 1rem;
        color: #616161;
    }
    .recommendation {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        border-left: 4px solid #43a047;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'expenses' not in st.session_state:
    # Try to load existing data
    try:
        if os.path.exists('expenses_data.json'):
            with open('expenses_data.json', 'r') as f:
                st.session_state.expenses = pd.read_json(f)
                # Convert date column to datetime
                st.session_state.expenses['date'] = pd.to_datetime(st.session_state.expenses['date'])
        else:
            st.session_state.expenses = pd.DataFrame(columns=[
                'date', 'amount', 'category', 'subcategory', 'description'
            ])
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.session_state.expenses = pd.DataFrame(columns=[
            'date', 'amount', 'category', 'subcategory', 'description'
        ])

# Function to save data
def save_data():
    st.session_state.expenses.to_json('expenses_data.json')

# Function to create download link for file export
def get_download_link(df, filename, text):
    if filename.endswith('.csv'):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    elif filename.endswith('.xlsx'):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Expenses')
        excel_data = output.getvalue()
        b64 = base64.b64encode(excel_data).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Function to generate recommendations based on spending data
def generate_recommendations(df):
    if df.empty:
        return ["Start tracking your expenses to get personalized recommendations."]
    
    recommendations = []
    
    # Calculate total spend by category for current month
    current_month = datetime.now().month
    current_year = datetime.now().year
    current_month_data = df[(df['date'].dt.month == current_month) & (df['date'].dt.year == current_year)]
    
    if not current_month_data.empty:
        category_spend = current_month_data.groupby('category')['amount'].sum().sort_values(ascending=False)
        
        # Top spending category recommendation
        top_category = category_spend.index[0]
        top_amount = category_spend.iloc[0]
        recommendations.append(f"Your highest spending category this month is {top_category} (₹{top_amount:.2f}). Consider setting a budget for this category.")
        
        # Look at top subcategory within top category
        if top_category in CATEGORIES:
            subcategory_data = current_month_data[current_month_data['category'] == top_category]
            if not subcategory_data.empty:
                subcategory_spend = subcategory_data.groupby('subcategory')['amount'].sum().sort_values(ascending=False)
                if not subcategory_spend.empty:
                    top_subcategory = subcategory_spend.index[0]
                    top_sub_amount = subcategory_spend.iloc[0]
                    
                    if top_category == "Food" and top_subcategory in ["Dining Out", "Food Delivery"]:
                        recommendations.append(f"You spent ₹{top_sub_amount:.2f} on {top_subcategory}. Consider cooking at home more often to save money.")
                    
                    elif top_category == "Transport" and top_subcategory in ["Cab/Auto", "Fuel"]:
                        recommendations.append(f"Your {top_subcategory} expenses (₹{top_sub_amount:.2f}) are high. Consider carpooling or using public transport when possible.")
                    
                    elif top_category == "Entertainment" or (top_category == "Shopping" and top_subcategory == "Clothing"):
                        recommendations.append(f"Your {top_subcategory} spending (₹{top_sub_amount:.2f}) is significant. Consider implementing a 24-hour rule before making non-essential purchases.")
                    
                    elif top_category == "Utilities":
                        recommendations.append(f"To reduce your {top_subcategory} bill (₹{top_sub_amount:.2f}), try to be more mindful of your usage and consider energy-efficient alternatives.")
        
        # Analyze frequency of small purchases
        small_purchases = current_month_data[current_month_data['amount'] < 100]
        if len(small_purchases) > 10:  # Arbitrary threshold for "many small purchases"
            total_small = small_purchases['amount'].sum()
            recommendations.append(f"You made {len(small_purchases)} small purchases (<₹100) totaling ₹{total_small:.2f}. These small expenses add up quickly!")
    
    # Compare with previous month if data available
    last_month = (datetime.now().replace(day=1) - timedelta(days=1)).month
    last_month_year = (datetime.now().replace(day=1) - timedelta(days=1)).year
    
    last_month_data = df[(df['date'].dt.month == last_month) & (df['date'].dt.year == last_month_year)]
    if not last_month_data.empty and not current_month_data.empty:
        last_month_total = last_month_data['amount'].sum()
        current_month_total = current_month_data['amount'].sum()
        
        if current_month_total > last_month_total:
            percent_increase = ((current_month_total - last_month_total) / last_month_total) * 100
            recommendations.append(f"Your spending is up {percent_increase:.1f}% compared to last month. Review your expenses to identify areas to cut back.")
        else:
            percent_decrease = ((last_month_total - current_month_total) / last_month_total) * 100
            if percent_decrease > 0:
                recommendations.append(f"Great job! Your spending is down {percent_decrease:.1f}% compared to last month.")
    
    # General recommendations if we don't have enough data
    if len(recommendations) < 2:
        recommendations.append("Set up automatic transfers to a savings account on payday to ensure you save before you spend.")
        recommendations.append("Consider using the 50-30-20 rule: 50% on needs, 30% on wants, and 20% on savings and debt repayment.")
    
    return recommendations

# Main app layout
def main():
    st.markdown('<h1 class="main-header">Personal Finance Tracker</h1>', unsafe_allow_html=True)
    
    # Sidebar menu
    st.sidebar.title("Menu")
    page = st.sidebar.radio("Navigate to", ["Dashboard", "Add Expense", "View Expenses", "Analysis", "Export Data"])
    
    if page == "Dashboard":
        display_dashboard()
    elif page == "Add Expense":
        add_expense()
    elif page == "View Expenses":
        view_expenses()
    elif page == "Analysis":
        analyze_expenses()
    elif page == "Export Data":
        export_data()

# Dashboard page
def display_dashboard():
    st.markdown('<h2 class="sub-header">Dashboard</h2>', unsafe_allow_html=True)
    
    # If no data, show welcome message
    if st.session_state.expenses.empty:
        st.markdown("""
        <div class="card">
            <h3>Welcome to your Personal Finance Tracker!</h3>
            <p>Start by adding your expenses in the "Add Expense" section.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Get current month data
    current_month = datetime.now().month
    current_year = datetime.now().year
    current_month_name = calendar.month_name[current_month]
    
    df = st.session_state.expenses
    current_month_data = df[(df['date'].dt.month == current_month) & (df['date'].dt.year == current_year)]
    
    # Calculate key metrics
    total_expense = current_month_data['amount'].sum() if not current_month_data.empty else 0
    avg_daily = current_month_data.groupby(current_month_data['date'].dt.day)['amount'].sum().mean() if not current_month_data.empty else 0
    category_totals = current_month_data.groupby('category')['amount'].sum().nlargest(3) if not current_month_data.empty else pd.Series()
    
    # Display key metrics
    st.markdown(f'<h3>Summary for {current_month_name} {current_year}</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">₹{total_expense:,.2f}</div>
            <div class="metric-label">Total Spent</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">₹{avg_daily:,.2f}</div>
            <div class="metric-label">Average Daily</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if not current_month_data.empty:
            top_category = category_totals.index[0] if not category_totals.empty else "None"
            top_amount = category_totals.iloc[0] if not category_totals.empty else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{top_category}</div>
                <div class="metric-label">Top Category (₹{top_amount:,.2f})</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">N/A</div>
                <div class="metric-label">Top Category</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Visualizations
    st.markdown("""<h3>This Month's Overview</h3>""", unsafe_allow_html=True)
    
    if not current_month_data.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Category breakdown
            fig = px.pie(
                current_month_data, 
                values='amount', 
                names='category',
                title='Spending by Category',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Daily spending
            daily_spending = current_month_data.groupby(current_month_data['date'].dt.day)['amount'].sum().reset_index()
            daily_spending.columns = ['day', 'amount']
            
            fig = px.bar(
                daily_spending,
                x='day',
                y='amount',
                title='Daily Spending',
                labels={'amount': 'Amount (₹)', 'day': 'Day of Month'},
                color_discrete_sequence=['#1E88E5']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Recent transactions
    st.markdown("""<h3>Recent Transactions</h3>""", unsafe_allow_html=True)
    
    if not df.empty:
        recent = df.sort_values('date', ascending=False).head(5)
        recent_formatted = recent.copy()
        recent_formatted['date'] = recent_formatted['date'].dt.strftime('%d %b %Y')
        recent_formatted['amount'] = recent_formatted['amount'].apply(lambda x: f"₹{x:,.2f}")
        
        # Use st.dataframe instead of st.table for better formatting
        st.dataframe(
            recent_formatted[['date', 'category', 'subcategory', 'amount', 'description']],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No transactions recorded yet.")
    
    # Recommendations
    st.markdown("""<h3>Smart Recommendations</h3>""", unsafe_allow_html=True)
    
    recommendations = generate_recommendations(df)
    for rec in recommendations:
        st.markdown(f"""
        <div class="recommendation">
            {rec}
        </div>
        """, unsafe_allow_html=True)

# Add expense page
def add_expense():
    st.markdown('<h2 class="sub-header">Add New Expense</h2>', unsafe_allow_html=True)
    
    with st.form("expense_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            date = st.date_input("Date", datetime.now())
            
            category = st.selectbox("Category", list(CATEGORIES.keys()))
            
            if category:
                subcategories = CATEGORIES[category]
                subcategory = st.selectbox("Subcategory", subcategories)
            else:
                subcategory = None
        
        with col2:
            amount = st.number_input("Amount (₹)", min_value=0.0, format="%.2f")
            description = st.text_area("Description (Optional)")
        
        submitted = st.form_submit_button("Add Expense")
        
        if submitted:
            if amount <= 0:
                st.error("Please enter a valid amount.")
            else:
                new_expense = pd.DataFrame({
                    'date': [date],
                    'amount': [amount],
                    'category': [category],
                    'subcategory': [subcategory],
                    'description': [description]
                })
                
                st.session_state.expenses = pd.concat([st.session_state.expenses, new_expense], ignore_index=True)
                save_data()
                
                st.success("Expense added successfully!")
                st.balloons()

# View expenses page
def view_expenses():
    st.markdown('<h2 class="sub-header">View Expenses</h2>', unsafe_allow_html=True)
    
    if st.session_state.expenses.empty:
        st.info("No expenses recorded yet. Go to 'Add Expense' to get started.")
        return
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Get min and max dates from data
        min_date = st.session_state.expenses['date'].min().date()
        max_date = st.session_state.expenses['date'].max().date()
        
        # Default to current month
        default_start = datetime(max_date.year, max_date.month, 1).date()
        default_end = max_date
        
        start_date = st.date_input("From", default_start, min_value=min_date, max_value=max_date)
    
    with col2:
        end_date = st.date_input("To", default_end, min_value=min_date)
    
    with col3:
        all_categories = list(CATEGORIES.keys())
        all_categories.insert(0, "All Categories")
        filter_category = st.selectbox("Category", all_categories)
    
    # Filter data
    filtered_data = st.session_state.expenses.copy()
    
    # Apply date filter
    filtered_data = filtered_data[(filtered_data['date'].dt.date >= start_date) & (filtered_data['date'].dt.date <= end_date)]
    
    # Apply category filter
    if filter_category != "All Categories":
        filtered_data = filtered_data[filtered_data['category'] == filter_category]
    
    # Display total for filtered data
    total_filtered = filtered_data['amount'].sum()
    st.markdown(f"""
    <div class="metric-card" style="margin-bottom: 20px;">
        <div class="metric-value">₹{total_filtered:,.2f}</div>
        <div class="metric-label">Total for Selected Period</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show data
    if not filtered_data.empty:
        # Format for display
        display_data = filtered_data.copy()
        display_data['date'] = display_data['date'].dt.strftime('%d %b %Y')
        display_data['amount'] = display_data['amount'].apply(lambda x: f"₹{x:,.2f}")
        
        # Sort by date (newest first)
        display_data = display_data.sort_values('date', ascending=False)
        
        # Display table
        st.dataframe(
            display_data[['date', 'category', 'subcategory', 'amount', 'description']],
            use_container_width=True,
            hide_index=True
        )
        
        # Add delete functionality
        if st.button("Delete Selected Expenses"):
            st.session_state.expenses = st.session_state.expenses[~st.session_state.expenses.index.isin(filtered_data.index)]
            save_data()
            st.success("Selected expenses deleted successfully!")
            st.experimental_rerun()
    else:
        st.info("No expenses found with the selected filters.")

# Analysis page
def analyze_expenses():
    st.markdown('<h2 class="sub-header">Expense Analysis</h2>', unsafe_allow_html=True)
    
    if st.session_state.expenses.empty:
        st.info("No expenses recorded yet. Go to 'Add Expense' to get started.")
        return
    
    # Time period selection
    analysis_period = st.selectbox(
        "Select Analysis Period",
        ["Current Month", "Last Month", "Last 3 Months", "Last 6 Months", "This Year", "All Time"]
    )
    
    # Filter data based on selected period
    df = st.session_state.expenses.copy()
    current_date = datetime.now()
    
    if analysis_period == "Current Month":
        df = df[(df['date'].dt.month == current_date.month) & (df['date'].dt.year == current_date.year)]
        period_name = f"{calendar.month_name[current_date.month]} {current_date.year}"
    elif analysis_period == "Last Month":
        last_month = current_date.month - 1 if current_date.month > 1 else 12
        last_month_year = current_date.year if current_date.month > 1 else current_date.year - 1
        df = df[(df['date'].dt.month == last_month) & (df['date'].dt.year == last_month_year)]
        period_name = f"{calendar.month_name[last_month]} {last_month_year}"
    elif analysis_period == "Last 3 Months":
        three_months_ago = current_date - timedelta(days=90)
        df = df[df['date'] >= three_months_ago]
        period_name = f"Last 3 Months"
    elif analysis_period == "Last 6 Months":
        six_months_ago = current_date - timedelta(days=180)
        df = df[df['date'] >= six_months_ago]
        period_name = f"Last 6 Months"
    elif analysis_period == "This Year":
        df = df[df['date'].dt.year == current_date.year]
        period_name = f"{current_date.year}"
    else:  # All Time
        period_name = "All Time"
    
    if df.empty:
        st.info(f"No expenses recorded for {period_name}.")
        return
    
    st.markdown(f"<h3>Analysis for {period_name}</h3>", unsafe_allow_html=True)
    
    # Key metrics
    total_spent = df['amount'].sum()
    avg_daily = df.groupby(df['date'].dt.date)['amount'].sum().mean()
    max_daily = df.groupby(df['date'].dt.date)['amount'].sum().max()
    max_single = df['amount'].max()
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">₹{total_spent:,.2f}</div>
            <div class="metric-label">Total Spent</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">₹{avg_daily:,.2f}</div>
            <div class="metric-label">Avg. Daily Spend</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">₹{max_daily:,.2f}</div>
            <div class="metric-label">Max Daily Spend</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">₹{max_single:,.2f}</div>
            <div class="metric-label">Largest Expense</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Visualization tabs
    tabs = st.tabs(["Category Analysis", "Time Analysis", "Subcategory Breakdown", "Day of Week Analysis"])
    
    with tabs[0]:
        # Category analysis
        st.subheader("Spending by Category")
        
        # Calculate and sort categories by amount
        category_totals = df.groupby('category')['amount'].sum().sort_values(ascending=False)
        
        # Bar chart for categories
        fig = px.bar(
            x=category_totals.index,
            y=category_totals.values,
            labels={'x': 'Category', 'y': 'Amount (₹)'},
            color_discrete_sequence=['#1E88E5'],
            text=category_totals.values.round(2)
        )
        
        fig.update_traces(texttemplate='₹%{text:,.2f}', textposition='outside')
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        st.plotly_chart(fig, use_container_width=True)
        
        # Category percentage
        st.subheader("Category Distribution")
        
        fig = px.pie(
            values=category_totals.values,
            names=category_totals.index,
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hoverinfo='label+percent+value'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        # Time analysis
        st.subheader("Spending Over Time")
        
        # Daily spending
        daily_spend = df.groupby(df['date'].dt.date)['amount'].sum().reset_index()
        daily_spend.columns = ['date', 'amount']
        
        # Line chart
        fig = px.line(
            daily_spend,
            x='date',
            y='amount',
            labels={'date': 'Date', 'amount': 'Amount (₹)'},
            markers=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Cumulative spending
        st.subheader("Cumulative Spending")
        
        daily_spend_sorted = daily_spend.sort_values('date')
        daily_spend_sorted['cumulative'] = daily_spend_sorted['amount'].cumsum()
        
        fig = px.line(
            daily_spend_sorted,
            x='date',
            y='cumulative',
            labels={'date': 'Date', 'cumulative': 'Cumulative Amount (₹)'},
            color_discrete_sequence=['#43a047']
        )
        
        # Add target line if period is current month or this year
        if analysis_period in ["Current Month", "This Year"]:
            current_day = current_date.day
            days_in_month = calendar.monthrange(current_date.year, current_date.month)[1]
            
            if analysis_period == "Current Month":
                # Calculate projected spending by end of month
                current_total = daily_spend_sorted['cumulative'].iloc[-1] if len(daily_spend_sorted) > 0 else 0
                projected = current_total * (days_in_month / current_day)
                
                # Add projection line
                x_projection = [daily_spend_sorted['date'].min(), current_date.date(), daily_spend_sorted['date'].max()]
                y_projection = [0, current_total, projected]
                
                fig.add_trace(
                    go.Scatter(
                        x=x_projection,
                        y=y_projection,
                        mode='lines',
                        line=dict(dash='dash', color='red'),
                        name='Projected'
                    )
                )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly comparison if we have enough data
        if len(df['date'].dt.to_period('M').unique()) > 1:
            st.subheader("Monthly Comparison")
            
            monthly_spend = df.groupby(df['date'].dt.to_period('M'))['amount'].sum().reset_index()
            monthly_spend['date'] = monthly_spend['date'].astype(str)
            
            fig = px.bar(
                monthly_spend,
                x='date',
                y='amount',
                labels={'date': 'Month', 'amount': 'Amount (₹)'},
                color_discrete_sequence=['#5e35b1'],
                text=monthly_spend['amount'].round(2)
            )
            
            fig.update_traces(texttemplate='₹%{text:,.2f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        # Subcategory breakdown
        st.subheader("Subcategory Analysis")
        
        # Select category to analyze
        selected_category = st.selectbox(
            "Choose Category for Detailed Analysis",
            category_totals.index
        )
        
        if selected_category:
            # Filter data for selected category
            category_data = df[df['category'] == selected_category]
            
            subcategory_totals = category_data.groupby('subcategory')['amount'].sum().sort_values(ascending=False)
            
            # Bar chart for subcategories
            fig = px.bar(
                x=subcategory_totals.index,
                y=subcategory_totals.values,
                labels={'x': 'Subcategory', 'y': 'Amount (₹)'},
                color_discrete_sequence=['#26a69a'],
                text=subcategory_totals.values.round(2)
            )
            
            fig.update_traces(texttemplate='₹%{text:,.2f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            
            # Top expenses in this category
            st.subheader(f"Top {selected_category} Expenses")
            
            top_expenses = category_data.sort_values('amount', ascending=False).head(5)
            top_expenses_display = top_expenses.copy()
            top_expenses_display['date'] = top_expenses_display['date'].dt.strftime('%d %b %Y')
            top_expenses_display['amount'] = top_expenses_display['amount'].apply(lambda x: f"₹{x:,.2f}")
            
            st.dataframe(
                top_expenses_display[['date', 'subcategory', 'amount', 'description']],
                use_container_width=True,
                hide_index=True
            )
            
    
    with tabs[3]:
        # Day of week analysis
        st.subheader("Spending by Day of Week")
        
        # Add day of week
        day_data = df.copy()
        day_data['day_of_week'] = day_data['date'].dt.day_name()
        
        # Order days correctly
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Calculate average spend by day of week
        day_totals = day_data.groupby('day_of_week')['amount'].sum().reindex(day_order)
        day_counts = day_data.groupby('day_of_week')['date'].nunique().reindex(day_order)
        day_avg = (day_totals / day_counts).fillna(0)
        
        # Bar chart for day of week
        fig = px.bar(
            x=day_avg.index,
            y=day_avg.values,
            labels={'x': 'Day of Week', 'y': 'Average Spending (₹)'},
            color_discrete_sequence=['#fb8c00'],
            text=day_avg.values.round(2)
        )
        
        fig.update_traces(texttemplate='₹%{text:,.2f}', textposition='outside')
        fig.update_layout(xaxis={'categoryorder': 'array', 'categoryarray': day_order})
        st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap of day of week vs category
        st.subheader("Day of Week vs Category Heatmap")
        
        # Create pivot table
        pivot_data = day_data.pivot_table(
            index='day_of_week',
            columns='category',
            values='amount',
            aggfunc='sum',
            fill_value=0
        ).reindex(day_order)
        
        # Create heatmap
        fig = px.imshow(
            pivot_data,
            labels=dict(x="Category", y="Day of Week", color="Amount (₹)"),
            x=pivot_data.columns,
            y=pivot_data.index,
            color_continuous_scale="Blues"
        )
        
        fig.update_layout(
            xaxis={'side': 'top'},
            coloraxis_colorbar=dict(title="Amount (₹)")
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations based on analysis
    st.markdown("<h3>Smart Recommendations</h3>", unsafe_allow_html=True)
    
    recommendations = generate_recommendations(df)
    for rec in recommendations:
        st.markdown(f"""
        <div class="recommendation">
            {rec}
        </div>
        """, unsafe_allow_html=True)

# Export data page
def export_data():
    st.markdown('<h2 class="sub-header">Export Your Data</h2>', unsafe_allow_html=True)
    
    if st.session_state.expenses.empty:
        st.info("No expenses recorded yet. Go to 'Add Expense' to get started.")
        return
    
    st.markdown("""
    <div class="card">
        <h3>Download Options</h3>
        <p>Export your expense data in various formats for offline analysis or backup.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h4>CSV Format</h4>', unsafe_allow_html=True)
        csv_data = st.session_state.expenses.copy()
        csv_data['date'] = csv_data['date'].dt.strftime('%Y-%m-%d')
        
        csv_link = get_download_link(csv_data, "finance_tracker_data.csv", "Download CSV")
        st.markdown(csv_link, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h4>Excel Format</h4>', unsafe_allow_html=True)
        excel_data = st.session_state.expenses.copy()
        excel_data['date'] = excel_data['date'].dt.strftime('%Y-%m-%d')
        
        excel_link = get_download_link(excel_data, "finance_tracker_data.xlsx", "Download Excel")
        st.markdown(excel_link, unsafe_allow_html=True)
    
    # Date range export
    st.markdown("<h4>Export Data for Specific Date Range</h4>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        min_date = st.session_state.expenses['date'].min().date()
        max_date = st.session_state.expenses['date'].max().date()
        
        start_date = st.date_input("From Date", min_date, key="export_start")
    
    with col2:
        end_date = st.date_input("To Date", max_date, key="export_end")
    
    if start_date and end_date:
        if start_date > end_date:
            st.error("Start date cannot be after end date.")
        else:
            # Filter data for selected date range
            filtered_data = st.session_state.expenses[
                (st.session_state.expenses['date'].dt.date >= start_date) & 
                (st.session_state.expenses['date'].dt.date <= end_date)
            ]
            
            filtered_data = filtered_data.copy()
            filtered_data['date'] = filtered_data['date'].dt.strftime('%Y-%m-%d')
            
            if not filtered_data.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_link = get_download_link(filtered_data, f"finance_data_{start_date}_to_{end_date}.csv", "Download Filtered CSV")
                    st.markdown(csv_link, unsafe_allow_html=True)
                
                with col2:
                    excel_link = get_download_link(filtered_data, f"finance_data_{start_date}_to_{end_date}.xlsx", "Download Filtered Excel")
                    st.markdown(excel_link, unsafe_allow_html=True)
            else:
                st.info("No data available for the selected date range.")
    
    # Data visualization export
    st.markdown("<h4>Export Visualizations</h4>", unsafe_allow_html=True)
    
    if not st.session_state.expenses.empty:
        # Generate some visualizations for export
        
        # Monthly spending bar chart
        monthly_data = st.session_state.expenses.copy()
        monthly_data['month'] = monthly_data['date'].dt.strftime('%Y-%m')
        monthly_spend = monthly_data.groupby('month')['amount'].sum().reset_index()
        
        fig1 = px.bar(
            monthly_spend,
            x='month',
            y='amount',
            title='Monthly Spending',
            labels={'month': 'Month', 'amount': 'Amount (₹)'},
            color_discrete_sequence=['#1E88E5']
        )
        
        # Category pie chart
        category_totals = st.session_state.expenses.groupby('category')['amount'].sum()
        
        fig2 = px.pie(
            values=category_totals.values,
            names=category_totals.index,
            title='Spending by Category',
            hole=0.4
        )
        
        fig2.update_traces(textposition='inside', textinfo='percent+label')
        
        # Display and offer download
        st.markdown("<p>Preview of visualizations available for export:</p>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(fig1, use_container_width=True)
            buf1 = io.BytesIO()
            fig1.write_image(buf1, format="png", width=800, height=400)
            btn1 = st.download_button(
                label="Download Monthly Chart",
                data=buf1.getvalue(),
                file_name="monthly_spending.png",
                mime="image/png"
            )
        
        with col2:
            st.plotly_chart(fig2, use_container_width=True)
            buf2 = io.BytesIO()
            fig2.write_image(buf2, format="png", width=800, height=400)
            btn2 = st.download_button(
                label="Download Category Chart",
                data=buf2.getvalue(),
                file_name="category_spending.png",
                mime="image/png"
            )

# Run the main app
if __name__ == "__main__":
    main()
