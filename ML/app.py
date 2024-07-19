import streamlit as st
import cloudpickle
import joblib
import numpy as np
import pandas as pd

# Load the model and encoders using cloudpickle
with open('le_car_value.pkl', 'rb') as f:
    le_car_value = cloudpickle.load(f)

with open('le_state_code.pkl', 'rb') as f:
    le_state_code = cloudpickle.load(f)

# Load joblib files for scalers
car_age_scaler = joblib.load('car_age_scaler.joblib')
age_youngest_scaler = joblib.load('age_youngest_scaler.joblib')

# Load the model
model = joblib.load('model.joblib')

def preprocess_input(data):
    # Convert inputs to DataFrame
    df = pd.DataFrame(data, columns=['state_code', 'group_size', 'homeowner', 'car_age', 'car_value', 
                                     'age_youngest', 'married_couple', 'c_previous', 'duration_previous', 
                                     'A', 'B', 'C', 'D', 'E', 'F', 'G'])
    
    # Extract state code from input
    state_code_input = df['state_code'].values[0]
    
    # Check if the state code is in the known classes before transformation
    if state_code_input not in le_state_code.classes_:
        # Handle the case of an unseen state code
        st.error(f"State code '{state_code_input}' is not recognized.")
        return None
    
    # Perform label encoding on the extracted state code
    df['state_code'] = le_state_code.transform([state_code_input])[0]
    
    # Convert categorical features to numeric using label encoders
    df['car_value'] = le_car_value.transform([df['car_value'].values[0]])  # Transform single value
    
    # Scale the specific columns
    df['car_age'] = car_age_scaler.transform([[df['car_age'].values[0]]])[0][0]
    df['age_youngest'] = age_youngest_scaler.transform([[df['age_youngest'].values[0]]])[0][0]
    
    # Convert to numpy array for model prediction
    scaled_data = df.to_numpy()
    
    return scaled_data

def main():
    st.title('Insurance Cost Prediction')
    st.markdown('Enter the required details to predict the insurance cost for your car:')
    
    state_codes = [
        "FL - Florida", "NY - New York", "PA - Pennsylvania", "OH - Ohio", "MD - Maryland",
        "IN - Indiana", "WA - Washington", "CO - Colorado", "AL - Alabama", "CT - Connecticut",
        "TN - Tennessee", "KY - Kentucky", "NV - Nevada", "MO - Missouri", "OR - Oregon",
        "UT - Utah", "OK - Oklahoma", "MS - Mississippi", "AR - Arkansas", "WI - Wisconsin",
        "GA - Georgia", "NH - New Hampshire", "NM - New Mexico", "ME - Maine", "ID - Idaho",
        "RI - Rhode Island", "KS - Kansas", "WV - West Virginia", "IA - Iowa", "DE - Delaware",
        "DC - District of Columbia", "MT - Montana", "NE - Nebraska", "ND - North Dakota",
        "WY - Wyoming", "SD - South Dakota"
    ]
    
    car_value = ["a - Below $10,001",
                 "b - $10,001 - $20,000",
                 "c - $20,001 - $30,000",
                 "d - $30,001 - $40,000",
                 "e - $40,001 - $50,000",
                 "f - $50,001 - $75,000",
                 "g - $75,001 - $1,00,000",
                 "h - $1,00,000 - $1,50,000",
                 "i = $1,50,001 and above"]
    c_previous = ["0 - None", "1 - Economy or Compact Car", "2 - Mid-sized Sedan or SUV", "3 - Luxury Vehicle", "4 - High-performance or Specialty Vehicle"]
    A_options = ["0 - Basic", "1 - Standard", "2 - Premium"]
    B_options = ["0 - Non Smoker", "1 - Smoker"]
    C_options = ["1 - Economy or Compact Car", "2 - Mid-sized Sedan or SUV", "3 - Luxury Vehicle", "4 - High-performance or Specialty Vehicle"]
    D_options = ["1 - Personal Use", "2 - Business Use (e.g., Commuting)", "3 - Commercial Use (e.g., Delivery or Rideshare)"]
    E_options = ["0 - Vehicle lacks specific safety features (e.g., Airbags, Anti-lock)", "1 - Vehicle is equipped with recommended safety features"]
    F_options = ["0 - Driver has a clean record with no accidents or violations", 
                 "1 - Driver has minor violations (e.g., Speeding Ticket)", 
                 "2 - Driver has a history of accidents or more serious violations", 
                 "3 - Driver has multiple accidents or severe violations"]
    G_options = ["1 - Urban or densely populated area with higher traffic and theft risk.", 
                 "2 - Suburban area with moderate traffic and lower theft risk.", 
                 "3 - Rural area with lower traffic and minimal theft risk", 
                 "4 - Hazardous or extreme risk zone (e.g., Flood-prone, High Crime)"]
    
    # Input fields
    state_code = st.selectbox('State Code', ['Please select a state'] + state_codes, index=0)
    group_size = st.text_input('Group Size (1-4)', placeholder='Enter the size of your group between 1 and 4')
    homeowner = st.text_input('Homeowner', placeholder='Enter 0 for No or 1 for Yes')
    car_age = st.text_input('How Old is your Car', placeholder='Enter the age of your car between 0 and 100')
    car_value = st.selectbox('Value of your Car at the Time of Purchase', ["Please select your car's value when you puchased it"] + car_value, index=0)    
    age_youngest = st.text_input('Age of the Youngest Person in your Group', placeholder="Enter the youngest person's between 16 and 80")
    married_couple = st.text_input('Married Couple', placeholder='Enter 0 for No or 1 for Yes')
    c_previous = st.selectbox('Previous Car Type', ['Please select your previous car type'] + c_previous, index=0)
    duration_previous = st.text_input('Duration with your Previous Insurer', placeholder='Enter a value between 0 and 15 in years')
    A = st.selectbox('Coverage Level', ['Please select the coverage level'] + A_options, index=0)
    B = st.selectbox('Smoker/Non-Smoker', ["Please select the policyholder's smoking type"] + B_options, index=0)
    C = st.selectbox('Car Type', ['Please select a car type'] + C_options, index=0)
    D = st.selectbox('Purpose of the Vehicle', ['Please select the purpose of the vehicle'] + D_options, index=0)
    E = st.selectbox('Safety Features', ['Please select the safety features'] + E_options, index=0)
    F = st.selectbox("Driver's Historic Record", ["Please select the driver's historic record"] + F_options, index=0)
    G = st.selectbox('Geographical Location', ['Please select the geographical location'] + G_options, index=0)
    
    # Extract state code number
    selected_state_code = state_code  # Store the full selected state code
    if state_code != 'Please select a state':
        state_code = state_code.split()[0]  # Modify state_code for validation
    else:
        st.error('Please select a state code.')
        return
    
    # Adjusting placeholder color for select boxes
    st.markdown(
        """
        <style>
        ::placeholder {
            color: lightgray;
            opacity: 1; /* Firefox */
        }

        :-ms-input-placeholder {
            color: lightgray;
        }

        ::-ms-input-placeholder {
            color: lightgray;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Encode selected options directly
    def encode_selected_option(option):
        return option.split()[0]

    car_value = encode_selected_option(car_value)
    c_previous = encode_selected_option(c_previous)
    A = encode_selected_option(A)
    B = encode_selected_option(B)
    C = encode_selected_option(C)
    D = encode_selected_option(D)
    E = encode_selected_option(E)
    F = encode_selected_option(F)
    G = encode_selected_option(G)
    
    # Validation function
    def validate_input(value, expected_type, min_value=None, max_value=None):
        if expected_type == 'int':
            try:
                int_val = int(value)
                if min_value is not None and int_val < min_value:
                    return False
                if max_value is not None and int_val > max_value:
                    return False
                return True
            except ValueError:
                return False
        elif expected_type == 'str':
            if value not in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']:
                return False
            return True  # Return True explicitly for string type
        elif expected_type == 'selectbox':
            if value.startswith('Please'):
                return False
            return True
        else:
            return False

    if st.button('Predict'):
        if state_code == 'Please select a state' or not state_code.isalpha() or len(state_code) != 2:
            st.error('Please select a valid State Code.')
            return
        if not validate_input(group_size, 'int', 1, 4):
            st.error('Group Size must be an integer between 1 and 4.')
            return
        if not validate_input(homeowner, 'int', 0, 1):
            st.error('Homeowner must be 0 or 1.')
            return
        if not validate_input(car_age, 'int', 0, 100):
            st.error('Car Age must be a value between 0 and 100.')
            return
        if not validate_input(car_value, 'str'):
            st.error('Car Value must be one of the following: a, b, c, d, e, f, g, h, i.')
            return
        # if not validate_input(risk_factor, 'int', 1, 4):
        #     st.error('Risk Factor must be an integer between 1 and 4.')
        #     return
        if not validate_input(age_youngest, 'int', 16, 100):
            st.error('Age of the youngest person must be a value between 16 and 100.')
            return
        if not validate_input(married_couple, 'int', 0, 1):
            st.error('Married Couple must be 0 or 1.')
            return
        if not validate_input(c_previous, 'selectbox'):
            st.error('Please select a valid Coverage Level.')
            return
        if not validate_input(duration_previous, 'int', 0, 15):
            st.error('Duration Previous must be an integer between 0 and 15.')
            return
        if not validate_input(A, 'selectbox'):
            st.error('Please select a valid Coverage Level.')
            return
        if not validate_input(B, 'selectbox'):
            st.error('Please select a valid Smoker/Non-Smoker status.')
            return
        if not validate_input(C, 'selectbox'):
            st.error('Please select a valid Car Type.')
            return
        if not validate_input(D, 'selectbox'):
            st.error('Please select a valid Purpose of the Vehicle.')
            return
        if not validate_input(E, 'selectbox'):
            st.error('Please select a valid Safety Features status.')
            return
        if not validate_input(F, 'selectbox'):
            st.error('Please select a valid Driver\'s Historic Record status.')
            return
        if not validate_input(G, 'selectbox'):
            st.error('Please select a valid Geographical Location.')
            return

        # Encode selected options directly
        def encode_selected_option(option):
            return option.split()[0]

        car_value = encode_selected_option(car_value)
        c_previous = encode_selected_option(c_previous)
        A = encode_selected_option(A)
        B = encode_selected_option(B)
        C = encode_selected_option(C)
        D = encode_selected_option(D)
        E = encode_selected_option(E)
        F = encode_selected_option(F)
        G = encode_selected_option(G)

        data = [[state_code, group_size, homeowner, car_age, car_value, 
                 age_youngest, married_couple, c_previous, duration_previous, 
                 A, B, C, D, E, F, G]]

        scaled_data = preprocess_input(data)
        
        if scaled_data is not None:
            # Make prediction
            prediction = model.predict(scaled_data)
            
            st.success(f'The predicted cost of insurance is: ${prediction[0]:.2f}')

if __name__ == '__main__':
    main()