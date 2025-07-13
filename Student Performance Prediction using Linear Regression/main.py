#!/usr/bin/env python
# coding: utf-8

# ## Step 1: Import Libraries

# In[64]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --------------------------
# Step 1: Load Dataset
# --------------------------
df = pd.read_csv('Student_Performance.csv')

# Step 2: Preprocess
if df['Extracurricular Activities'].dtype == 'object':
    le = LabelEncoder()
    df['Extracurricular Activities'] = le.fit_transform(df['Extracurricular Activities'])

# Step 3: Split Features/Target
x = df.drop(columns='Performance Index')
y = df['Performance Index']

# Step 4: Train/Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Step 5: Scale Features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Step 6: Train Model
model = LinearRegression()
model.fit(x_train_scaled, y_train)

# Step 7: Evaluate
y_pred = model.predict(x_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Step 8: Save model and scaler
joblib.dump(model, 'performance_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# --------------------------
# Streamlit App UI
# --------------------------

# Load model and scaler
model = joblib.load('performance_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🎓 Student Performance Predictor")

# Input Fields
col1, col2 = st.columns(2)

with col1:
    hours = st.number_input("📚 Hours Studied", min_value=0.0)
    prev_scores = st.number_input("📝 Previous Scores (out of 100)", min_value=0.0)

with col2:
    sleep = st.number_input("😴 Sleep Hours per Day", min_value=0.0)
    papers = st.number_input("📄 Practice Papers Solved", min_value=0)

activity = st.selectbox("🎭 Extracurricular Activities", ['Yes', 'No'])

# Predict Button
if st.button("🔮 Predict Performance Index"):

    # Input validation
    if hours == 0 or prev_scores == 0 or sleep == 0 or papers == 0:
        st.warning("⚠️ Please fill all fields with non-zero values for an accurate prediction.")
    else:
        activity_encoded = 1 if activity == 'Yes' else 0
        input_data = np.array([[hours, prev_scores, sleep, activity_encoded, papers]])
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]

        st.success(f"🎯 Predicted Performance Index: {prediction:.2f}")

        # Optional range (you can adjust margin based on std of residuals if desired)
        st.info(f"📉 Estimated Range: {prediction - 5:.2f} to {prediction + 5:.2f}")

# --------------------------
# Visualization Section
# --------------------------

st.markdown("---")
st.header("📊 Model Evaluation")

st.write(f"**Mean Squared Error:** {mse:.2f}")
st.write(f"**R² Score:** {r2:.2f}")

# Feature Importance
st.subheader("📌 Feature Importance")
coef_df = pd.DataFrame(model.coef_, index=x.columns, columns=['Coefficient'])
st.bar_chart(coef_df.sort_values(by='Coefficient'))

# Actual vs Predicted Plot
st.subheader("🎯 Actual vs Predicted")

fig1, ax1 = plt.subplots()
ax1.scatter(y_test, y_pred, alpha=0.7)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax1.set_xlabel("Actual Performance Index")
ax1.set_ylabel("Predicted Performance Index")
ax1.set_title("Actual vs Predicted")
ax1.grid(True)
st.pyplot(fig1)

# Residual Plot
st.subheader("🔎 Residuals Plot")
residuals = y_test - y_pred

fig2, ax2 = plt.subplots()
ax2.scatter(y_pred, residuals, color='purple', alpha=0.6)
ax2.axhline(y=0, color='red', linestyle='--')
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Residuals")
ax2.set_title("Residual Plot")
ax2.grid(True)
st.pyplot(fig2)

# Highlight Top 5 Largest Errors
abs_errors = abs(y_test - y_pred)
top5_idx = abs_errors.sort_values(ascending=False).head(5).index
top5_pos = [y_test.index.get_loc(i) for i in top5_idx]

fig3, ax3 = plt.subplots()
ax3.scatter(y_test, y_pred, alpha=0.6, label='All Points')
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', label='Perfect Prediction')

# ✅ Fixed scatter line
ax3.scatter(y_test.loc[top5_idx], y_pred[top5_pos], color='orange', edgecolors='black', s=100, label='Top 5 Errors')

# ✅ Add labels for error points
for i, pos in zip(top5_idx, top5_pos):
    ax3.text(y_test[i], y_pred[pos] + 1, f"{i}", fontsize=9)

ax3.set_xlabel("Actual")
ax3.set_ylabel("Predicted")
ax3.set_title("Top 5 Largest Errors")
ax3.legend()
ax3.grid(True)
st.pyplot(fig3)



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




