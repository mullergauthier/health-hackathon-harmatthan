import streamlit as st
import pandas as pd

# ----- Sidebar -----
with st.sidebar:
    st.button("Logo")  # Placeholder for logo
    st.button("Direct")
    st.button("Language")
    st.button("LLM 1")
    st.button("LLM 2")

# ----- Main Panel -----
st.title("Doctor Input Interface")

# Doctor input section
doctor_notes = st.text_area("Doctor input (notes)", height=150)

# Extracted diseases display
st.subheader("Extracted diseases")
st.markdown("*(This area will show extracted disease names based on the input above)*")
extracted_placeholder = st.empty()

# Simulated extracted diseases (placeholder)
extracted_diseases = ["Diabetes", "Hypertension"]
with extracted_placeholder.container():
    for disease in extracted_diseases:
        st.write(f"- {disease}")

# ---- Output Table Section ----
st.subheader("Output")

# Placeholder for a dynamic editable table (simulated)
output_data = [
    {"Extract": "Diabetes", "Code 1": "E10", "Desc 1": "Type 1 Diabetes", "Code 2": "E11", "Desc 2": "Type 2 Diabetes"},
    {"Extract": "Hypertension", "Code 1": "I10", "Desc 1": "Essential Hypertension", "Code 2": "", "Desc 2": ""},
]

# Display table with actions
for i, row in enumerate(output_data):
    cols = st.columns([2, 1, 2, 1, 2, 0.5, 0.5])
    cols[0].text_input("Extract", value=row["Extract"], key=f"extract_{i}")
    cols[1].text_input("Code 1", value=row["Code 1"], key=f"code1_{i}")
    cols[2].text_input("Desc 1", value=row["Desc 1"], key=f"desc1_{i}")
    cols[3].text_input("Code 2", value=row["Code 2"], key=f"code2_{i}")
    cols[4].text_input("Desc 2", value=row["Desc 2"], key=f"desc2_{i}")
    delete = cols[5].button("❌", key=f"delete_{i}")
    confirm = cols[6].button("✅", key=f"confirm_{i}")

    if delete:
        st.warning(f"Row {i+1} marked for deletion.")
    if confirm:
        st.success(f"Row {i+1} confirmed.")

