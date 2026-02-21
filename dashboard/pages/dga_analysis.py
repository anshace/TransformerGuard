"""
DGA Analysis Page
Interactive DGA interpretation tool with multiple diagnosis methods
"""

# Import dashboard components
import sys
from datetime import datetime
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

dashboard_path = Path(__file__).parent.parent
sys.path.insert(0, str(dashboard_path))

from components.duval_plot import calculate_duval_percentages, create_duval_triangle
from components.gas_chart import create_gas_bar_chart

# API Configuration
API_BASE_URL = "http://localhost:8000"


def analyze_dga_manual(
    h2: float,
    ch4: float,
    c2h2: float,
    c2h4: float,
    c2h6: float,
    co: float,
    co2: float,
    transformer_id: int = 1,
) -> dict:
    """Send DGA data to API for analysis."""
    try:
        data = {
            "transformer_id": transformer_id,
            "sample_date": datetime.utcnow().isoformat(),
            "h2": h2,
            "ch4": ch4,
            "c2h2": c2h2,
            "c2h4": c2h4,
            "c2h6": c2h6,
            "co": co,
            "co2": co2,
            "method": "multi",
        }
        response = requests.post(
            f"{API_BASE_URL}/api/v1/dga/analyze", json=data, timeout=30
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"API error: {str(e)}")
    return None


def analyze_dga_file(file, transformer_id: int) -> dict:
    """Upload and analyze DGA file."""
    try:
        # Read file content
        content = file.read()

        # Determine file type and parse
        if file.name.endswith(".csv"):
            df = pd.read_csv(StringIO(content.decode("utf-8")))
        elif file.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(content)
        else:
            st.error("Unsupported file format. Please use CSV or Excel.")
            return None

        # Process the data
        results = []
        for _, row in df.iterrows():
            result = analyze_dga_manual(
                h2=float(row.get("h2", 0)),
                ch4=float(row.get("ch4", 0)),
                c2h2=float(row.get("c2h2", 0)),
                c2h4=float(row.get("c2h4", 0)),
                c2h6=float(row.get("c2h6", 0)),
                co=float(row.get("co", 0)),
                co2=float(row.get("co2", 0)),
                transformer_id=transformer_id,
            )
            if result:
                results.append(result)

        return {"results": results, "total": len(results)}
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
    return None


def main():
    """Main DGA analysis page."""
    st.title("🔬 DGA Analysis")
    st.markdown(
        "Dissolved Gas Analysis interpretation tool with multiple diagnosis methods"
    )

    # Input method selector
    st.sidebar.markdown("### Input Method")
    input_method = st.sidebar.radio(
        "Select input method:", ["Manual Entry", "File Upload", "Select Transformer"]
    )

    # Initialize variables
    dga_data = None
    transformer_id = 1

    if input_method == "Manual Entry":
        st.markdown("### Manual Gas Input")
        st.info("Enter gas concentrations in ppm (parts per million)")

        # Gas input form
        with st.form("dga_input_form"):
            col1, col2 = st.columns(2)

            with col1:
                h2 = st.number_input(
                    "Hydrogen (H2)", min_value=0.0, value=0.0, step=1.0
                )
                ch4 = st.number_input(
                    "Methane (CH4)", min_value=0.0, value=0.0, step=1.0
                )
                c2h2 = st.number_input(
                    "Acetylene (C2H2)", min_value=0.0, value=0.0, step=1.0
                )
                c2h4 = st.number_input(
                    "Ethylene (C2H4)", min_value=0.0, value=0.0, step=1.0
                )

            with col2:
                c2h6 = st.number_input(
                    "Ethane (C2H6)", min_value=0.0, value=0.0, step=1.0
                )
                co = st.number_input(
                    "Carbon Monoxide (CO)", min_value=0.0, value=0.0, step=1.0
                )
                co2 = st.number_input(
                    "Carbon Dioxide (CO2)", min_value=0.0, value=0.0, step=1.0
                )

            submit_button = st.form_submit_button("Analyze DGA")

        if submit_button:
            with st.spinner("Analyzing DGA..."):
                dga_data = analyze_dga_manual(h2, ch4, c2h2, c2h4, c2h6, co, co2)

    elif input_method == "File Upload":
        st.markdown("### File Upload")

        uploaded_file = st.file_uploader(
            "Upload DGA data file (CSV or Excel)", type=["csv", "xlsx", "xls"]
        )

        if uploaded_file:
            transformer_id_input = st.number_input(
                "Target Transformer ID", min_value=1, value=1
            )

            if st.button("Analyze File"):
                with st.spinner("Processing file..."):
                    dga_data = analyze_dga_file(uploaded_file, transformer_id_input)

    elif input_method == "Select Transformer":
        st.markdown("### Select Transformer")

        # Fetch transformer list
        try:
            response = requests.get(
                f"{API_BASE_URL}/api/v1/transformers?limit=100", timeout=10
            )
            if response.status_code == 200:
                transformers = response.json()

                transformer_options = {t["id"]: t["name"] for t in transformers}
                selected_id = st.selectbox(
                    "Select Transformer",
                    options=list(transformer_options.keys()),
                    format_func=lambda x: transformer_options[x],
                )

                if st.button("Use Latest DGA"):
                    # Fetch latest DGA
                    try:
                        resp = requests.get(
                            f"{API_BASE_URL}/api/v1/dga/{selected_id}", timeout=10
                        )
                        if resp.status_code == 200:
                            dga_data = resp.json()
                            transformer_id = selected_id
                    except:
                        st.error("Could not fetch DGA data")
        except:
            st.error("Could not fetch transformers")

    # Display analysis results
    if dga_data:
        # Check if it's a batch result
        if "results" in dga_data:
            st.markdown(f"### Analysis Results ({dga_data.get('total', 0)} samples)")

            for i, result in enumerate(dga_data.get("results", [])):
                st.markdown(f"#### Sample {i + 1}")
                dga_data = result
                # Continue to display this result
        else:
            st.markdown("### Analysis Results")

        # Display fault diagnosis
        st.markdown("#### Fault Diagnosis")

        if isinstance(dga_data, dict):
            fault_type = dga_data.get("fault_type", "Unknown")
            fault_confidence = dga_data.get("fault_confidence", 0)
            explanation = dga_data.get("explanation", "No explanation available")

            # Fault type with color coding
            fault_colors = {
                "PD": "#3498db",
                "D1": "#e74c3c",
                "D2": "#c0392b",
                "T1": "#f39c12",
                "T2": "#e67e22",
                "T3": "#d35400",
                "DT": "#9b59b6",
                "Normal": "#2ecc71",
            }

            fault_color = fault_colors.get(fault_type.split()[0], "#95a5a6")

            st.markdown(
                f"**Fault Type:** <span style='color:{fault_color}; font-size:18px; font-weight:bold'>{fault_type}</span>",
                unsafe_allow_html=True,
            )

            st.metric("Confidence", f"{fault_confidence * 100:.1f}%")
            st.write(f"**Explanation:** {explanation}")

            # Get gas values
            h2 = dga_data.get("h2", 0)
            ch4 = dga_data.get("ch4", 0)
            c2h2 = dga_data.get("c2h2", 0)
            c2h4 = dga_data.get("c2h4", 0)
            c2h6 = dga_data.get("c2h6", 0)
            co = dga_data.get("co", 0)
            co2 = dga_data.get("co2", 0)

            # Calculate TDCG
            tdcg = h2 + ch4 + c2h2 + c2h4 + c2h6 + co + co2
            st.metric("Total Dissolved Combustible Gas (TDCG)", f"{tdcg:.1f} ppm")

            # Two-column layout for visualizations
            col1, col2 = st.columns(2)

            with col1:
                # Duval Triangle
                st.markdown("#### Duval Triangle")

                if ch4 and c2h4 and c2h2:
                    ch4_pct, c2h4_pct, c2h2_pct = calculate_duval_percentages(
                        ch4, c2h4, c2h2
                    )

                    fig = create_duval_triangle(ch4_pct, c2h4_pct, c2h2_pct, height=450)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Not enough gas data for Duval Triangle")

            with col2:
                # Gas distribution
                st.markdown("#### Gas Distribution")

                gas_df = pd.DataFrame(
                    {
                        "Gas": ["H2", "CH4", "C2H2", "C2H4", "C2H6", "CO", "CO2"],
                        "Value": [h2, ch4, c2h2, c2h4, c2h6, co, co2],
                    }
                )

                fig_bar = create_gas_bar_chart(
                    gas_df, "value", "Gas Concentrations", height=400
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            # Method comparison
            st.markdown("#### Method Comparison")

            method_results = dga_data.get("method_results", {})

            if method_results:
                method_data = []
                for method, result in method_results.items():
                    if isinstance(result, dict):
                        fault = result.get("fault_type", "Unknown")
                        confidence = result.get("fault_confidence", 0)
                        method_data.append(
                            {
                                "Method": method.upper(),
                                "Fault Type": fault,
                                "Confidence": f"{confidence * 100:.1f}%",
                            }
                        )

                if method_data:
                    method_df = pd.DataFrame(method_data)
                    st.table(method_df)
                else:
                    st.info("Method comparison not available")
            else:
                st.info("Detailed method results not available")

            # Additional info
            if dga_data.get("sample_date"):
                st.caption(f"Sample Date: {dga_data['sample_date'][:10]}")

    else:
        # Show placeholder content
        st.markdown("""
        ### How to Use DGA Analysis
        
        1. **Manual Entry**: Enter gas concentrations directly
        2. **File Upload**: Upload a CSV or Excel file with gas data
        3. **Select Transformer**: Use existing DGA data from a transformer
        
        ### Supported Gases
        - **Hydrogen (H2)**: Indicates partial discharge or thermal faults
        - **Methane (CH4)**: Indicates thermal faults
        - **Acetylene (C2H2)**: Indicates high-energy discharge
        - **Ethylene (C2H4)**: Indicates high-temperature thermal faults
        - **Ethane (C2H6)**: Indicates thermal faults
        - **Carbon Monoxide (CO)**: Indicates thermal degradation of cellulose
        - **Carbon Dioxide (CO2)**: Indicates aging of cellulose insulation
        """)

        # Show Duval Triangle legend
        st.markdown("### Duval Triangle Fault Zones")

        from components.duval_plot import create_duval_legend

        fig_legend = create_duval_legend()
        st.plotly_chart(fig_legend, use_container_width=True)


if __name__ == "__main__":
    main()
