import streamlit as st
import pandas as pd
from dss_engine import topsis, ahp_process, calculate_profile_matching_score, create_categorical_dataframe
import os

#############################################################################################################################################
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #f0f2f5;
    background-image: url("https://images.unsplash.com/photo-1648563643923-2091f9c0c12f?q=80&w=1931&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");}

</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

############################################################################################################################################
st.markdown("<center><h1>🖥️ Monitor Selection DSS 🖥️</h1></center>", unsafe_allow_html=True)


st.image("https://images.unsplash.com/photo-1614624532983-4ce03382d63d?q=80&w=1931&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
         caption="Select the best monitor for your needs using our Decision Support System powered by AHP, PM, and TOPSIS methods.",
         use_container_width=True,
         )

# --- User Input ---

# st.markdown("---")
st.header("🌟 Monitor Selection Criteria Hierarchy 🌟")
# Initialize session state
if "criteria_hierarchy" not in st.session_state:
    st.session_state.criteria_hierarchy = {
        'Display Quality': ['Refresh Rate', 'Resolution', 'Screen type'],
        'Design': ['Size', 'Weight'],
        'Value': ['Price', 'Warranty'],
        'Features': ['Features'],
    }

if "alternative_data_df" not in st.session_state:
    st.session_state.alternative_data_df = None
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None

criteria_hierarchy = st.session_state.criteria_hierarchy

### Criteria Hierarchy
if criteria_hierarchy:
    criteria_hierarchy_df = pd.DataFrame(list(criteria_hierarchy.items()), columns=['Category', 'Sub-Criteria'])
    st.write(criteria_hierarchy_df.set_index('Category'))
else:
    st.info("The criteria hierarchy is currently empty. Please add categories and sub-criteria.")

error_occurred = False

# Define input type and options per criterion
input_config = {
    "Refresh Rate": {"type": "number", "min": 30, "max": 480, "step": 1},
    "Resolution": {"type": "select", "options": ["HD (1280 x 720)", "Full HD (1920 x 1080)", "Ultra HD (2560 x 1440)", "Quad HD (3840 x 2160)", "8K (7680 x 4320)"]},
    "Screen type": {"type": "select", "options": ['VA', 'TN', 'IPS', 'OLED']},
    "Size": {"type": "number", "min": 11, "max": 50, "step": 1},
    "Weight": {"type": "number", "min": 1, "max": 50, "step": 1},
    "Price": {"type": "number", "min": 100000, "max": 10000000, "step": 100000},
    "Warranty": {"type": "number", "min": 0, "max": 10, "step": 1},
    "Features": {"type": "number", "min": 1, "max": 5, "step": 1},
}

st.markdown("---")
st.header("📝 Input Alternative Data 📝")

input_method = st.radio("Choose data input method:", ("Upload CSV", "Manual Input"))
if input_method == "Upload CSV":

    # File uploader and persistence of alternative data
    import_alternative_data = st.file_uploader("Import data from CSV", type="csv")
    if import_alternative_data is not None:
        try:
            flat_criteria = [sc for category in criteria_hierarchy.values() for sc in category]

            # Error handling: Check if there are any criteria to match with CSV columns
            if not flat_criteria:
                st.error("Cannot upload data. No criteria defined in the hierarchy. Please add categories and sub-criteria first.")
                error_occurred = True
            else:
                df_raw = pd.read_csv(import_alternative_data)
                # st.write("### Raw Data")
                # st.write(df_raw)
                df = create_categorical_dataframe(df_raw)
                if len(df.columns) != len(flat_criteria):
                    raise ValueError("Length mismatch: Uploaded data does not match expected number of sub-criteria.")
                df.columns = flat_criteria
                df.index = [f"Monitor {i+1}" for i in range(len(df))]
                st.session_state.alternative_data_df = df
                st.session_state.df_raw = df_raw # change the raw data to the session state
                st.success("Data successfully uploaded.")
                # st.write(df)
        except ValueError as e:
            error_occurred = True
            if "Length mismatch" in str(e):
                st.error("Column mismatch: Uploaded data does not match expected number of sub-criteria. Ensure the number of columns in your CSV matches the total number of sub-criteria defined.")
            else:
                st.error(f"An error occurred during data upload: {e}")
        except Exception as e:
            error_occurred = True
            st.error(f"An unexpected error occurred: {e}")

        if "alternative_data_df" in st.session_state and st.session_state.alternative_data_df is not None:
            st.write("### Uploaded Data")
            try:
                st.write(st.session_state.df_raw.set_index(st.session_state.df_raw.columns[0]))
            except Exception:
                st.write(st.session_state.alternative_data_df)

if input_method == "Manual Input":
    
    st.title("📺 Multi-Monitor Data Entry Form")

    # CSV file path
    CSV_FILE = "monitors.csv"

    # Field definitions
    fields = {
        "Monitor Model": {"type": "text"},
        "Refresh Rate": {"type": "number", "min": 30, "max": 480, "step": 1},
        "Resolution": {"type": "select", "options": ["HD (1280 x 720)", "Full HD (1920 x 1080)", "Ultra HD (2560 x 1440)", "Quad HD (3840 x 2160)", "8K (7680 x 4320)"]},
        "Screen type": {"type": "select", "options": ['VA', 'TN', 'IPS', 'OLED']},
        "Size": {"type": "number", "min": 11, "max": 50, "step": 1},
        "Weight": {"type": "number", "min": 1, "max": 50, "step": 1},
        "Price": {"type": "number", "min": 100000, "max": 10000000, "step": 100000},
        "Warranty": {"type": "number", "min": 0, "max": 10, "step": 1},
        "Features": {"type": "number", "min": 1, "max": 5, "step": 1},
    }

    # Number of monitors input
    num_monitors = st.number_input("How many monitors do you want to input?", min_value=1, max_value=20, step=1)

    # Begin form
    with st.form("monitor_batch_form"):
        all_monitor_data = []
        st.subheader("Enter Monitor Data")

        for i in range(num_monitors):
            with st.expander(f"🖥️ Monitor {i + 1} Details", expanded=(i == 0)):
                monitor_data = {}

                # Use 2 columns layout
                cols = st.columns(3)  # You can use 2 or 3 depending on how compact you want it

                for idx, (field, config) in enumerate(fields.items()):
                    with cols[idx % 3]:  # Distribute fields across columns
                        key = f"{field}_{i}"
                        if config["type"] == "number":
                            monitor_data[field] = st.number_input(
                                label=field,
                                min_value=config["min"],
                                max_value=config["max"],
                                step=config["step"],
                                key=key
                            )
                        elif config["type"] == "select":
                            monitor_data[field] = st.selectbox(
                                label=field,
                                options=config["options"],
                                key=key
                            )
                        else:
                            monitor_data[field] = st.text_input(label=field, key=key)

                all_monitor_data.append(monitor_data)

        submitted = st.form_submit_button("Submit All Monitors")

        if submitted:
            if os.path.exists(CSV_FILE):
                df = pd.read_csv(CSV_FILE)
            else:
                df = pd.DataFrame(columns=fields.keys())

            df_new = pd.DataFrame(all_monitor_data)
            df = pd.concat([df, df_new], ignore_index=True)
            df.to_csv(CSV_FILE, index=False)
            st.success(f"{num_monitors} monitor(s) successfully added to {CSV_FILE}!")

    # Display current data
    if os.path.exists(CSV_FILE):
        st.subheader("📄 Current Monitor Dataset")
        st.dataframe(pd.read_csv(CSV_FILE).set_index("Monitor Model"))
        
        flat_criteria = [sc for category in criteria_hierarchy.values() for sc in category]
        df_raw = pd.read_csv(CSV_FILE)
        df = create_categorical_dataframe(df_raw)
        if len(df.columns) != len(flat_criteria):
            raise ValueError("Length mismatch: Uploaded data does not match expected number of sub-criteria.")
        df.columns = flat_criteria
        df.index = [f"Monitor {i+1}" for i in range(len(df))] # just add a alternative for the new index
        st.session_state.alternative_data_df = df
        st.success("Data successfully uploaded.")

        # Add a button to delete the CSV file
        if st.button("Remove CSV File"):
            os.remove(CSV_FILE)
            st.info("CSV file removed.")
    else:
        st.info("No data available yet.")
    
# Display the final processed alternative data table
if st.session_state.alternative_data_df is not None:
    alternative_data = st.session_state.alternative_data_df.to_dict(orient='records')


#####################################################################################################################################
st.markdown("---")
st.header("⚖️Criteria Weighting with AHP⚖️")
st.subheader("Main Criteria Pairwise Comparison")

main_criteria = list(criteria_hierarchy.keys())
main_criteria_matrix = {}

# Add an option to import a CSV file
import_csv = st.file_uploader("Import pairwise comparison matrix from CSV", type="csv")
if import_csv is not None:
    try:
        criteria_hierarchy_df = pd.read_csv(import_csv)
        criteria_hierarchy_df.index = criteria_hierarchy_df.columns
        criteria_hierarchy_df.columns = criteria_hierarchy_df.columns
        st.write(criteria_hierarchy_df)
        main_criteria_matrix = criteria_hierarchy_df.to_numpy()
        
        ## Check consistency
        main_weights, main_cr = ahp_process(main_criteria_matrix)

        st.write(f"##### Main Criteria Weights:")
        for i, criterion in enumerate(main_criteria):
            st.write(f"{criterion}: {main_weights[i]:.3f}")

        st.write(f"\nConsistency Ratio: {main_cr:.3f}")

        if main_cr > 0.1:
            st.warning("The consistency ratio is high. Please check your pairwise comparisons.")
        else:
            st.success("The pairwise comparisons are consistent.")
    except ValueError as e:
        if "Length mismatch" in str(e):
            st.error("Length mismatch error: The number of columns in the uploaded CSV file does not match the expected number of main criteria.")
        else:
            st.error(f"An error occurred: {e}")
else:
    # Loop through each main criterion
    for i, main in enumerate(main_criteria):
        # Create a subheader for the current main criterion
        st.subheader(f"{main} Pairwise Comparison")
        
        # Initialize an empty list to store the comparison values
        comparison_values = []
        
        # Loop through each other main criterion for comparison
        for j in range(i + 1, len(main_criteria)):
            # Create a selectbox to choose the comparison preference
            preference = st.selectbox(f"{main} vs {main_criteria[j]}", [ 'Extremely worse', 'Very strongly worse', 'Strongly worse', 'Moderately worse', 'Equal', 'Moderate', 'Strong', 'Very Strong', 'Extreme'], index=4)
            
            # Map the preference to a numerical value
            if preference == 'Equal':
                value = 1
            elif preference == 'Moderate':
                value = 2
            elif preference == 'Strong':
                value = 3
            elif preference == 'Very Strong':
                value = 4
            elif preference == 'Extreme':
                value = 5
            elif preference == 'Moderately worse':
                value = 1/2
            elif preference == 'Strongly worse':
                value = 1/3
            elif preference == 'Very strongly worse':
                value = 1/4
            elif preference == 'Extremely worse':
                value = 1/5
                
            comparison_values.append(value)
        main_criteria_matrix[main] = comparison_values

    # Create the full matrix by adding the reciprocal values
    full_matrix = []
    for i, main in enumerate(main_criteria):
        row = []
        for j in range(len(main_criteria)):
            if i == j:
                row.append(1)  # Identity comparison
            elif i < j:
                row.append(main_criteria_matrix[main][j - i - 1])  # Use the stored value
            else:
                row.append(1 / main_criteria_matrix[main_criteria[j]][i - j - 1])  # Use the reciprocal value
        full_matrix.append(row)
    
    # Print the resulting matrix
    st.write("Main Criteria Pairwise Comparison Matrix:")
    full_matrix_df = pd.DataFrame(full_matrix, index=main_criteria, columns=main_criteria)
    st.write(full_matrix_df)

    ## Check consistency
    main_weights, main_cr = ahp_process(full_matrix)

    st.write("Main Criteria Weights:")
    for i, criterion in enumerate(main_criteria):
        st.write(f"{criterion}: {main_weights[i]:.3f}")

    st.write(f"\nConsistency Ratio: {main_cr:.3f}")

    if main_cr > 0.1:
        st.warning("The consistency ratio is high. Please check your pairwise comparisons.")
    else:
        st.success("The pairwise comparisons are consistent.")

# Initialize a dictionary to store all sub-criteria matrices
all_sub_criteria_matrices = {}

for main in main_criteria:
    st.write(f"##### {main} Sub-Criteria Pairwise Comparison")
    
    # Get the sub-criteria for the current main criterion
    sub_criteria = criteria_hierarchy[main]
    
    # Initialize an empty dictionary to store the sub-criteria comparison values
    sub_criteria_matrix = {}
    
    # Loop through each sub-criterion
    for i, sub in enumerate(sub_criteria):
        # Initialize an empty list to store the comparison values
        comparison_values = []
        
        # Loop through each other sub-criterion for comparison
        for j in range(i + 1, len(sub_criteria)):
            # Create a selectbox to choose the comparison preference
            preference = st.selectbox(
                f"{sub} vs {sub_criteria[j]} for {main}",  # Unique key by including main
                ['Extremely worse', 'Very strongly worse', 'Strongly worse', 'Moderately worse', 'Equal', 'Moderate', 'Strong', 'Very Strong', 'Extreme'],
                index=4,
                key=f"{main}_{sub}_{sub_criteria[j]}"  # Unique key for Streamlit
            )
            
            # Map the preference to a numerical value
            if preference == 'Equal':
                value = 1
            elif preference == 'Moderate':
                value = 2
            elif preference == 'Strong':
                value = 3
            elif preference == 'Very Strong':
                value = 4
            elif preference == 'Extreme':
                value = 5
            elif preference == 'Moderately worse':
                value = 1/2
            elif preference == 'Strongly worse':
                value = 1/3
            elif preference == 'Very strongly worse':
                value = 1/4
            elif preference == 'Extremely worse':
                value = 1/5
                
            comparison_values.append(value)
        sub_criteria_matrix[sub] = comparison_values
    
    # Build the full sub-criteria matrix
    full_sub_criteria_matrix = []
    for i, sub in enumerate(sub_criteria):
        row = []
        for j in range(len(sub_criteria)):
            if i == j:
                row.append(1)  # Identity comparison
            elif i < j:
                row.append(sub_criteria_matrix[sub][j - i - 1])  # Use the stored value
            else:
                row.append(1 / sub_criteria_matrix[sub_criteria[j]][i - j - 1])  # Use the reciprocal value
        full_sub_criteria_matrix.append(row)
    
    # Store the matrix in the dictionary
    all_sub_criteria_matrices[main] = full_sub_criteria_matrix
    
    # Display the resulting matrix
    st.write(f"{main} Sub-Criteria Pairwise Comparison Matrix:")
    # st.dataframe(full_sub_criteria_matrix)
    sub_criteria_df = pd.DataFrame(full_sub_criteria_matrix, index=sub_criteria, columns=sub_criteria)
    st.dataframe(sub_criteria_df)
    
    # Check consistency
    sub_criteria_weights, sub_criteria_cr = ahp_process(full_sub_criteria_matrix)
    
    st.write(f"##### {main} Sub-Criteria Weights:")
    for i, sub in enumerate(sub_criteria):
        st.write(f"{sub}: {sub_criteria_weights[i]:.3f}")
    
    st.write(f"###### {main} Sub-Criteria Consistency Ratio: {sub_criteria_cr:.3f}")
    
    # Check if the consistency ratio is high
    if sub_criteria_cr > 0.1:
        st.warning(f"The consistency ratio for {main} sub-criteria is high. Please check your pairwise comparisons.")
    else:
        st.success(f"The pairwise comparisons for {main} sub-criteria are consistent.")


# Calculate local and global weights for subcriteria
local_weights = {}
global_weights = {}

for criterion, subcriteria in criteria_hierarchy.items():
    # Get main criterion weight
    main_weight = main_weights[main_criteria.index(criterion)]
    
    # Check if the main criterion has a subcriteria matrix
    if criterion in all_sub_criteria_matrices:
        # Calculate local weights for subcriteria
        matrix = all_sub_criteria_matrices[criterion]
        weights, cr = ahp_process(matrix)
        
        # Store local weights and calculate global weights
        local_weights[criterion] = {}
        for i, subcriterion in enumerate(subcriteria):
            local_weights[criterion][subcriterion] = weights[i]
            global_weights[subcriterion] = weights[i] * main_weight
    else:
        st.error(f"No subcriteria matrix found for main criterion '{criterion}'")

st.subheader("📊 Global Subcriteria Weights")

sorted_weights = sorted(global_weights.items(), key=lambda x: x[1], reverse=True)  # Optional: sorted by weight
markdown_output = "\n".join([f"- **{k}**: `{v:.3f}`" for k, v in sorted_weights])
st.markdown(markdown_output)

#############################################################################################################################################
st.markdown("---")
st.header("🎯Alternative Scoring with PM🎯") 

# Inputting ideal profile for each sub-criterion
st.subheader("Ideal Profile")
ideal_profiles = {}

ideal_profiles['Refresh Rate'] = st.slider('Refresh Rate (Hz)', 30, 480,180)
ideal_profiles['Resolution'] = st.selectbox('Resolution', ["HD (1280 x 720)", "Full HD (1920 x 1080)", "Ultra HD (2560 x 1440)", "Quad HD (3840 x 2160)", "8K (7680 x 4320)"])
ideal_profiles['Screen type'] = st.selectbox('Screen type', ['OLED', 'IPS', 'VA', 'TN'])
ideal_profiles['Size'] = st.slider('Size (Inches)', 10, 50, 27)
ideal_profiles['Weight'] = st.slider('Weight (Kg)', 1, 15, 4)
ideal_profiles['Price'] = st.number_input('Price (Rp.)', 1, 9999999999, 1500000, 100000)
ideal_profiles['Warranty'] = st.slider('Warranty (Years)', 0, 10, 3)
ideal_profiles['Features'] = st.slider('Features', 1, 5, 3)

ideal_profiles_df = pd.DataFrame(ideal_profiles, index=[0])
converted_ideal_profiles = create_categorical_dataframe(ideal_profiles_df)
dict_converted_ideal_profiles = converted_ideal_profiles.to_dict()

transposed_ideal_profiles = converted_ideal_profiles
# st.write(ideal_profiles)
ideal_profiles_dict = transposed_ideal_profiles.to_dict(orient='records')[0]
# st.write(ideal_profiles_dict)

# Define the gap weights
gap_weights = {
    0: 5,  # perfect match
    1: 4.5,  
    -1: 4.5,  
    2: 4,
    -2: 4,
    3: 3.5,
    -3: 3.5,  
    4: 3,
    -4: 3,  
    5: 2.5,
    -5: 2.5, 
}

if error_occurred:
    st.error("Please check your inputs.")
if 'alternative_data' in locals() and not error_occurred:
    st.caption("Ideal Profile")
    # st.write(ideal_profiles)
    ideal_profile_table = pd.DataFrame(list(ideal_profiles.items()), columns=['Subcriterion', 'Ideal Value'])
    st.write(ideal_profile_table.set_index('Subcriterion'))
    # st.caption("Gap Weights")
    gap_weights_df = pd.DataFrame(list(gap_weights.items()), columns=['Gap', 'Weight'])
    gap_weights_df = gap_weights_df.set_index('Gap')
    # st.write(gap_weights_df)
    # st.dataframe(alternative_data)
    
    # Calculate profile matching scores for each monitor on each subcriterion
    profile_scores = {}
    try:
        alternative_monitors = df_raw.iloc[:, 0]
    except (NameError, UnboundLocalError):
        st.error("An error occurred: The uploaded data is no longer available. Please refresh the page and re-upload your CSV file.")
        st.stop() 

    for i, specs in enumerate(alternative_data):
        profile_scores[alternative_monitors[i]] = {}
        
        for subcriterion, value in specs.items():
            ideal = ideal_profiles_dict[subcriterion]
            gap = value - ideal
            score = calculate_profile_matching_score(value, ideal, gap_weights)
            profile_scores[alternative_monitors[i]][subcriterion] = score

    # # Create a DataFrame for profile matching scores
    profile_scores_df = pd.DataFrame(profile_scores)
    # st.write("Profile Matching Scores for each Subcriterion:")
    # st.dataframe(profile_scores_df.T) 

    # Aggregate subcriteria scores to main criteria level using local weights
    aggregated_scores = {}

    for i, specs in enumerate(alternative_data):
        aggregated_scores[alternative_monitors[i]] = {}
        
        for main_criterion, subcriteria in criteria_hierarchy.items():
            weighted_sum = 0
            for subcriterion in subcriteria:
                weighted_sum += profile_scores[alternative_monitors[i]][subcriterion] * local_weights[main_criterion][subcriterion]
            
            aggregated_scores[alternative_monitors[i]][main_criterion] = weighted_sum

    # Create a DataFrame for aggregated scores
    aggregated_scores_df = pd.DataFrame(aggregated_scores)
    # st.write("\nAggregated Scores for each Main Criterion:")
    # st.dataframe(aggregated_scores_df.T)  


########################################################################################################################################
st.markdown("---")
st.header("🥇Final Ranking with TOPSIS🥇")

if st.button("Rank Monitors"):
    if input_method == "Upload CSV" and import_alternative_data is None and 'alternative_data' in locals():
        st.error("Please input alternative data to rank monitors.")
    elif 'profile_scores' in locals():
        # Connects Profile Matching with TOPSIS by using profile scores as input to TOPSIS
        profile_scores_matrix = pd.DataFrame(aggregated_scores).T

        # For TOPSIS with profile scores, all criteria become benefit criteria (higher score is better)
        criteria = list(profile_scores_matrix.columns)

        # Apply TOPSIS using global weights from AHP
        topsis_weights = pd.Series(global_weights)
        topsis_weights = pd.Series(main_weights, index=criteria)  # Ensure weights are aligned with criteria
        profile_topsis_result = topsis(profile_scores_matrix, topsis_weights)

        # Calculate rank
        profile_topsis_result['Rank'] = profile_topsis_result['Closeness'].rank(ascending=False).astype(int)
        
        st.write("Final Rank of Monitors (Using Profile Matching Scores):")
        st.write(profile_topsis_result[['Monitor', 'Closeness', 'Rank']].sort_values('Rank').set_index('Rank'))

        # Get the top-ranked monitor
        top_monitor = profile_topsis_result.loc[profile_topsis_result['Rank'] == 1, 'Monitor'].values[0]

        # Add conclusion sentence
        st.markdown(f"### ✅ Recommended Monitor\nBased on your selections, criteria, and ideal profile, our top recommendation is **{top_monitor}**.")

    else:
        st.error("Check your inputs.")

