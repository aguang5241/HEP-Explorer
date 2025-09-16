import json
import hashlib
import pandas as pd
import numpy as np
import streamlit as st
import torch
import pyro
import pickle
import plotly.express as px


torch.classes.__path__ = []

    
class BNN(pyro.nn.PyroModule):
    def __init__(self, in_features, out_features, n_layers, prior_scale_weight, prior_scale_bias, nodes_list):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.n_layers = n_layers
        self.prior_scale_weight = prior_scale_weight
        self.prior_scale_bias = prior_scale_bias
        self.nodes_list = nodes_list
        self.layers = torch.nn.ModuleList()
        self.activation = torch.nn.ReLU()

        # Get the suggested number of nodes for each layer
        self.nodes = [self.in_features]
        for i in range(self.n_layers):
            nodes = self.nodes_list[i]
            self.nodes.append(nodes)
        self.nodes.append(self.out_features)

        # Define Bayesian Linear layers
        layer_list = [pyro.nn.PyroModule[torch.nn.Linear](self.nodes[i-1], self.nodes[i]) for i in range(1, len(self.nodes))]
        self.layers = pyro.nn.PyroModule[torch.nn.ModuleList](layer_list)
        for i, layer in enumerate(self.layers):
            layer.weight = pyro.nn.PyroSample(pyro.distributions.Normal(0., self.prior_scale_weight).expand([self.nodes[i+1], self.nodes[i]]).to_event(2))
            layer.bias = pyro.nn.PyroSample(pyro.distributions.Normal(0., self.prior_scale_bias).expand([self.nodes[i+1]]).to_event(1))

    def forward(self, x, y=None):
        # Reshape the input
        x = x.reshape(-1, self.nodes[0])
        
        # Pass through hidden layers
        x = self.activation(self.layers[0](x))
        for layer in self.layers[1:-1]:
            x = self.activation(layer(x))
        mu = self.layers[-1](x)

        # Bayesian inference for output uncertainty
        sigma = pyro.sample('sigma', pyro.distributions.Gamma(0.5, 1.))

        # Observational likelihood
        with pyro.plate('data', x.shape[0]):
            obs = pyro.sample('obs', pyro.distributions.Normal(mu, sigma).to_event(1), obs=y)

        return mu


def BNN_predictor(data_pre):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the x and y scaler
    with open('res/scaler_x_BNN.pkl', 'rb') as f:
        x_scaler = pickle.load(f)
    with open('res/scaler_y_BNN.pkl', 'rb') as f:
        y_scaler = pickle.load(f)

    # Load the train data
    with open('res/train_x_BNN.pkl', 'rb') as f:
        train_x = pickle.load(f)
    with open('res/train_y_BNN.pkl', 'rb') as f:
        train_y = pickle.load(f)
    train_x = torch.FloatTensor(train_x).to(DEVICE)
    train_y = torch.FloatTensor(train_y).to(DEVICE)

    # Get the input data to predict
    x_pre = data_pre[['atomic_number_A','atomic_number_B','ionic_radius_A','ionic_radius_B','dopant_concentration_A','dopant_concentration_B']].to_numpy()
    x_pre = x_scaler.transform(x_pre)
    x_pre = torch.FloatTensor(x_pre).to(DEVICE)

    # Define the model
    model_path = 'res/model_BNN.pkl'
    IN_FEATURES = 6
    OUT_FEATURES = 3
    N_LAYERS = 2
    PRIOR_SCALE_WEIGHT = 1.1127440261121575
    PRIOR_SCALE_BIAS = 0.21609393343755173
    NODES_LIST = [26, 44]
    LR = 0.009945018994994233
    PREIDCTIVE_SAMPLES = 1000
    torch.manual_seed(0)
    model = BNN(in_features=IN_FEATURES, out_features=OUT_FEATURES, n_layers=N_LAYERS, prior_scale_weight=PRIOR_SCALE_WEIGHT, prior_scale_bias=PRIOR_SCALE_BIAS, nodes_list=NODES_LIST).to(DEVICE)
    # Define variational guide
    guide = pyro.infer.autoguide.AutoDiagonalNormal(model)
    # Optimizer
    optimizer = pyro.optim.Adam({'lr': LR})
    # Loss function
    loss_func = pyro.infer.Trace_ELBO()
    # Stochastic Variational Inference
    svi = pyro.infer.SVI(model, guide, optimizer, loss=loss_func)
    # Predictive
    predictive = pyro.infer.Predictive(model, guide=guide, num_samples=PREIDCTIVE_SAMPLES)
    # Load the model
    pyro.clear_param_store()
    from functools import partial
    torch.load = partial(torch.load, weights_only=False)
    pyro.get_param_store().load(model_path)
    pyro.module('model', model, update_module_params=True)
    # Activate the model
    preds_act = predictive(train_x, train_y)

    # Get the predicted data
    with torch.no_grad():
        preds = predictive(x_pre)
        y_pre = preds['obs'].mean(0).cpu().numpy()
        y_std = preds['obs'].std(0).cpu().numpy()
        y_pre = y_scaler.inverse_transform(y_pre)
        y_std = y_std * y_scaler.scale_
        data_pred_BNN = pd.concat([data_pre, pd.DataFrame(y_pre, columns=['F', 'lattice_distortion', 'atomic_distortion']), pd.DataFrame(y_std, columns=['F_std', 'lattice_distortion_std', 'atomic_distortion_std'])], axis=1)
        data_pred_BNN = data_pred_BNN[['atomic_number_A', 'atomic_number_B', 'ionic_radius_A', 'ionic_radius_B', 'dopant_concentration_A', 'dopant_concentration_B', 'oxygen_vacancy_concentration', 'T', 'F', 'lattice_distortion', 'atomic_distortion', 'F_std', 'lattice_distortion_std', 'atomic_distortion_std']]

    return data_pred_BNN


def get_dopant_data(dopant_A, dopant_B, dopant_A_conc, dopant_B_conc, oxygen_vacancy_concentration, T):
    # Define the atomic numbers and ionic radii for the dopants
    atomic_numbers = {'Mg': 12, 'Ca': 20, 'Sr': 38, 'Ba': 56, 'Ce': 58, 
                    'Pr': 59, 'Nd': 60, 'Sm': 62, 'Gd': 64, 'Sc': 21, 
                    'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 
                    'Ni': 28, 'Cu': 29, 'Zn': 30, 'Al': 13, 'Ga': 31}
    ionic_radii = {'Mg': 0.890, 'Ca': 1.340, 'Sr': 1.440, 'Ba': 1.610, 'Ce': 1.340, 
                'Pr': 1.179, 'Nd': 1.270, 'Sm': 1.240, 'Gd': 1.107, 'Sc': 0.745, 
                'Ti': 0.670, 'V': 0.640, 'Cr': 0.615, 'Mn': 0.645, 'Fe': 0.645, 
                'Ni': 0.600, 'Cu': 0.540, 'Zn': 0.740, 'Al': 0.535, 'Ga': 0.620}
    
    # Get the atomic number and ionic radius for A-site dopant
    if dopant_A != 'None':
        atomic_number_A = atomic_numbers[dopant_A]
        ionic_radius_A = ionic_radii[dopant_A]
        dopant_concentrations_A = np.arange(dopant_A_conc[0], dopant_A_conc[1] + 1, 1)
    else:
        atomic_number_A = 0
        ionic_radius_A = 0
        dopant_concentrations_A = [0]
    
    # Get the atomic number and ionic radius for B-site dopant
    if dopant_B != 'None':
        atomic_number_B = atomic_numbers[dopant_B]
        ionic_radius_B = ionic_radii[dopant_B]
        dopant_concentrations_B = np.arange(dopant_B_conc[0], dopant_B_conc[1] + 1, 1)
    else:
        atomic_number_B = 0
        ionic_radius_B = 0
        dopant_concentrations_B = [0]

    # Create a DataFrame to store the data
    data_pre = pd.DataFrame()
    for i, A_conc in enumerate(dopant_concentrations_A):
        for j, B_conc in enumerate(dopant_concentrations_B):
            data_row = pd.DataFrame({
                'atomic_number_A': [atomic_number_A],
                'atomic_number_B': [atomic_number_B],
                'ionic_radius_A': [ionic_radius_A],
                'ionic_radius_B': [ionic_radius_B],
                'dopant_concentration_A': [A_conc/100],
                'dopant_concentration_B': [B_conc/100],
                'oxygen_vacancy_concentration': [oxygen_vacancy_concentration/100],
                'T': [T],
            })
            data_pre = pd.concat([data_pre, data_row], ignore_index=True)

    return data_pre


def ANN_predictor(data_pred_BNN):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the scaler
    with open('res/scaler_x_ANN.pkl', 'rb') as f:
        x_scaler = pickle.load(f)
    with open('res/scaler_y_ANN.pkl', 'rb') as f:
        y_scaler = pickle.load(f)
    
    # Load the model
    model = torch.load('res/model_ANN.pkl', weights_only=False)
    
    # Get the input data to predict
    x_pre = data_pred_BNN[['oxygen_vacancy_concentration', 'T', 'F', 'lattice_distortion', 'atomic_distortion']].to_numpy()
    x_pre = x_scaler.transform(x_pre)
    x_pre = torch.FloatTensor(x_pre).to(DEVICE)
   
    # Predict the data
    with torch.no_grad():
        model.eval()
        y_pre = model(x_pre)
        y_pre = y_scaler.inverse_transform(y_pre)
    data_pred_ANN = data_pred_BNN.copy()
    data_pred_ANN['D'] = y_pre

    return data_pred_ANN


def D2C(D, T):
    z = 2                           # charge number of the Oxygen ion
    F = 9.64853399e4                # Faraday constant, C/mol
    R = 8.31446261815324            # J/(K・mol)
    C = z**2 * D * F**2 / (R*T*100) # S/cm
    return C


def data_display(data):
    data_dis = pd.DataFrame()
    data_dis['Dopant A'] = data['atomic_number_A'].map({12: 'Mg', 20: 'Ca', 38: 'Sr', 56: 'Ba', 58: 'Ce', 59: 'Pr', 60: 'Nd', 62: 'Sm', 64: 'Gd', 0: 'None'})
    data_dis['Dopant B'] = data['atomic_number_B'].map({21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 28: 'Ni', 29: 'Cu', 30: 'Zn', 13: 'Al', 31: 'Ga', 0: 'None'})
    data_dis['Dopant Concentration A (at.%)'] = (data['dopant_concentration_A'] * 100).round().astype(int)
    data_dis['Dopant Concentration B (at.%)'] = (data['dopant_concentration_B'] * 100).round().astype(int)
    data_dis['Oxygen Vacancy Concentration (at.%)'] = data['oxygen_vacancy_concentration'] * 100
    data_dis['Temperature (K)'] = data['T'].astype(int)
    data_dis['F (eV/atom)'] = data['F']
    data_dis['Lattice Distortion (%)'] = data['lattice_distortion'] * 100
    data_dis['Atomic Distortion (Å)'] = data['atomic_distortion']
    data_dis['F std (eV/atom)'] = data['F_std']
    data_dis['Lattice Distortion std (%)'] = data['lattice_distortion_std'] * 100
    data_dis['Atomic Distortion std (Å)'] = data['atomic_distortion_std']
    data_dis['D (cm^2/s)'] = data['D']
    data_dis['C (S/cm)'] = D2C(data_dis['D (cm^2/s)'], data_dis['Temperature (K)'])
    # Reorder the columns
    data_dis = data_dis[['Dopant A', 'Dopant Concentration A (at.%)', 'Dopant B', 'Dopant Concentration B (at.%)', 'Oxygen Vacancy Concentration (at.%)', 'Temperature (K)', 'F (eV/atom)', 'Lattice Distortion (%)', 'Atomic Distortion (Å)', 'F std (eV/atom)', 'Lattice Distortion std (%)', 'Atomic Distortion std (Å)', 'D (cm^2/s)', 'C (S/cm)']]
    return data_dis


def main():
    # Set the page config
    st.set_page_config(
        page_title='HEP-Explorer', 
        layout='wide', 
        page_icon=':material/apps:', 
        menu_items={
        'Get Help': 'https://github.com/aguang5241/HEP-Explorer',
        'Report a bug': 'mailto:gliu4@wpi.edu',
        'About': '# HEP-Explorer'
        '\n**AI Powered Materials Innovation**  '
        '\n\n*Developed by Guangchen Liu*  '
        '\n*IMPD Group, Worcester Polytechnic Institute, MA USA*',
    })
    
    # Add app logo to the sidebar
    st.sidebar.image('res/logo.png', width='stretch')
    # Set the sidebar title
    st.sidebar.title('HEP-Explorer  [![GitHub stars](https://img.shields.io/github/stars/aguang5241/HEP-Explorer?style=social)](https://github.com/aguang5241/HEP-Explorer)')
    # Add a description to the sidebar
    st.sidebar.markdown('An application for analyzing substitutional effects in high-entropy perovskites (HEPs), supporting composition optimization and material performance enhancement.', unsafe_allow_html=True)
    # Add a citation link to the sidebar
    st.sidebar.divider()
    st.sidebar.markdown('If you find this application useful, please consider citing our publication:')
    st.sidebar.markdown('[![DOI](https://img.shields.io/badge/DOI-xx.xxx/xxx--xxx--xxx--xxx-white?logo=doi&logoColor=white)](https://doi.org/)')
    # Add contact information: gliu4@wpi.edu
    # st.sidebar.divider()
    st.sidebar.markdown('For any questions or suggestions, please contact:')
    st.sidebar.markdown('[![Email](https://img.shields.io/badge/Email-yzhong@wpi.edu-white?logo=mail.ru&logoColor=white)](mailto:yzhong@wpi.edu)')
    st.sidebar.markdown('[![Email](https://img.shields.io/badge/Email-gliu4@wpi.edu-white?logo=mail.ru&logoColor=white)](mailto:gliu4@wpi.edu)')
    st.sidebar.markdown('[![LinkedIn](https://img.shields.io/badge/LinkedIn-Guangchen%20Liu-white?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/aguang5241)')
    
    # Add a title to the main page
    st.title('HEP-Explorer  [![GitHub stars](https://img.shields.io/github/stars/aguang5241/HEP-Explorer?style=social)](https://github.com/aguang5241/HEP-Explorer)')
    # Add a description to the main page
    st.markdown('An application for analyzing substitutional effects in high-entropy perovskites (HEPs), supporting composition optimization and material performance enhancement.', unsafe_allow_html=True)

    # Add a subheader for the substitutions selection
    st.divider()
    st.subheader('Substitutional Site:')
    # Select A-site or B-site substitutional site selection using segmented controls
    select_site = st.segmented_control(
        label='Please select the substitutional site in $ABO_3$ Perovskite:',
        options=['A-site', 'B-site'],
        default='A-site',
        width='stretch',
    )

    # Add multiselect for substitutional elements selection
    st.subheader('Substitutional Elements:')
    if select_site == 'A-site':
        element_pool = ['La', 'Ca', 'Sr', 'Ba', 'Ce', 'Pr', 'Nd', 'Sm', 'Gd']
        element = st.multiselect(
            label='Please select five A-site element(s) for substitution:',
            options=element_pool,
            default=['La', 'Ca', 'Sr', 'Ba', 'Ce'],
            max_selections=5,
        )
        # Ensure La is always included
        if 'La' not in element:
            st.error("Error: La is mandatory for A-site substitution.")
        # Ensure 5 elements are selected
        if len(element) != 5:
            st.error("Error: Please select exactly five elements.")
    else:
        element_pool = ['Co', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Cu', 'Zn']
        element = st.multiselect(
            label='Please select five B-site element(s) for substitution:',
            options=element_pool,
            default=['Co', 'Ni', 'Fe', 'Mn', 'Cr'],
            max_selections=5,
        )
        # Ensure Co is always included
        if 'Co' not in element:
            st.error("Error: Co is mandatory for B-site substitution.")
        # Ensure 5 elements are selected
        if len(element) != 5:
            st.error("Error: Please select exactly five elements.")

    # Add sliders for dopant concentration range selection
    st.subheader('Substitutional Concentrations:')
    st.caption('The concentration of La (for A-site) or Co (for B-site) will be adjusted to balance the total concentration to 100 at.%.')
    concentration_container = st.container(border=True, gap='medium', width='stretch')
    with concentration_container:
        if select_site == 'A-site':
            for i, elem in enumerate(element):
                if elem == 'La':
                    pass
                else:
                    conc = st.slider(
                        label=f'**{elem}** (at.%):',
                        min_value=15,
                        max_value=40,
                        value=(20, 30),
                        step=1,
                        key=f'slider_A_{i}',
                    )
                    element[i] = (elem, conc)
        else:
            for i, elem in enumerate(element):
                if elem == 'Co':
                    pass
                else:
                    conc = st.slider(
                        label=f'**{elem}** (at.%):',
                        min_value=15,
                        max_value=40,
                        value=(20, 30),
                        step=1,
                        key=f'slider_B_{i}',
                    )
                    element[i] = (elem, conc)

    # Add a container for system conditions
    st.subheader('System Conditions:')
    st.caption('This only applies to diffusivity and conductivity predictions.')
    temperature_container = st.container(border=True, gap='medium', width='stretch')
    
    # Set the 2nd column for temperature
    with temperature_container:
        T = st.slider(
            label='Please select the temperature (K):',
            min_value=500,
            max_value=2500,
            value=1000,
            step=10,
        )
        if T < 1000:
            st.warning('Warning: The model was trained on data above 1000 K, the prediction of diffusivity and conductivity may not be reliable at lower temperatures.')
        if T > 2000:
            st.warning('Warning: The model was trained on data below 2000 K, the prediction of diffusivity and conductivity may not be reliable at higher temperatures.')
    
    # Once the user dopant selection and system conditions changed, reset the session state for showing predicted data and visualization
    input_signature = json.dumps({
        "select_site": select_site,
        "element_conc": element,
        "temperature": T,
    }, sort_keys=True)
    st.write(f"Input Signature: {input_signature}")  # For debugging purpose only, can be removed later

    # Create a hash of the input state
    input_hash = hashlib.md5(input_signature.encode()).hexdigest()

    # Compare with the last stored input hash
    if 'last_input_hash' not in st.session_state:
        st.session_state['last_input_hash'] = input_hash
    else:
        if st.session_state['last_input_hash'] != input_hash:
            # Inputs have changed — reset prediction states
            st.session_state['data_pred_dis'] = None
            st.session_state['last_input_hash'] = input_hash

    # Add a button to submit the selections
    st.divider()
    if st.button('Predict', type='primary', width='stretch'):
        if select_site == 'A-site':
            if 'La' not in element:
                st.error("Error: La is mandatory for A-site substitution.")
                st.stop()
        else:
            if 'Co' not in element:
                st.error("Error: Co is mandatory for B-site substitution.")
                st.stop()
        # Ensure 5 elements are selected
        if len(element) != 5:
            st.error("Error: Please select exactly five elements.")
            st.stop()

        # Show a loading spinner
        with st.spinner('Predicting... Please wait — this may take a few seconds to a few minutes depending on your input size and complexity.'):
            try:
                # Get the dopant data
                data_pre_BNN = get_dopant_data(dopant_A, dopant_B, dopant_A_conc, dopant_B_conc, oxygen_vacancy_concentration, T)
                # Predict the data using BNN
                data_pred_BNN = BNN_predictor(data_pre_BNN)
                # Predict the data using ANN
                data_pred_ANN = ANN_predictor(data_pred_BNN)
                # Process the predicted data for display
                st.session_state['data_pred_dis'] = data_display(data_pred_ANN)
            except Exception as e:
                # Clear previous predictions
                st.session_state['data_pred_dis'] = None
                # Show user-friendly error message
                st.error('😢 Sorry, something unexpected happened. Please try again.')
                st.stop()
    
    # Show the predicted data
    if st.session_state.get('data_pred_dis') is not None:
        data_dis = st.session_state['data_pred_dis']

        # Visualize the forming energy
        st.divider()
        st.subheader('Energetic Stability:')
        with st.container(border=True):
            st.markdown('**Formation Energy (eV/atom):**')
            # Create a heatmap for F (eV/atom)
            if dopant_A != 'None' and dopant_B != 'None':
                heatmap_F = data_dis.pivot_table(index='Dopant Concentration A (at.%)', columns='Dopant Concentration B (at.%)', values='F (eV/atom)')
                labels=dict(x=f'{dopant_B} (at.%)', y=f'{dopant_A} (at.%)', color='F (eV/atom)')
            elif dopant_A != 'None' and dopant_B == 'None':
                heatmap_F = data_dis.pivot_table(index='Dopant Concentration B (at.%)', columns='Dopant Concentration A (at.%)', values='F (eV/atom)')
                labels=dict(x=f'{dopant_A} (at.%)', y='', color='F (eV/atom)')
            elif dopant_A == 'None' and dopant_B != 'None':
                heatmap_F = data_dis.pivot_table(index='Dopant Concentration A (at.%)', columns='Dopant Concentration B (at.%)', values='F (eV/atom)')
                labels=dict(x=f'{dopant_B} (at.%)', y='', color='F (eV/atom)')
            # Create the Plotly heatmap
            fig = px.imshow(
                heatmap_F,
                color_continuous_scale='Viridis',
                labels=labels,
            )
            fig.update_coloraxes(
                colorbar_tickformat='.2f',
                colorbar_title_text='',
            )
            # Display in Streamlit
            st.plotly_chart(fig, width='stretch')
        
        with st.expander('**Formation Energy Trends Across Different Dopant Sites** (Click to Expand)', expanded=False):
            if dopant_A != 'None' and dopant_B != 'None':
                # Set 2 columns for A and B site dopant
                col5, col6 = st.columns(2, border=True, gap='medium')
                with col5:
                    # Plot the F (eV/atom) vs B-site dopant concentration
                    if dopant_A_conc[0] == dopant_A_conc[1]:
                        dopant_A_conc_filter = dopant_A_conc[0]
                    else:
                        dopant_A_conc_filter = st.slider(
                            label=f'Adjust the Concentration of {dopant_A} (at.%):',
                            min_value=dopant_A_conc[0],
                            max_value=dopant_A_conc[1],
                            value=(dopant_A_conc[0] + dopant_A_conc[1]) // 2,
                            step=1,
                            key='dopant_A_conc_filter_for_F_A'
                        )
                    data_dis_filter = data_dis[data_dis['Dopant Concentration A (at.%)'] == dopant_A_conc_filter]
                    fig = px.line(
                        data_dis_filter,
                        x='Dopant Concentration B (at.%)',
                        y='F (eV/atom)',
                        error_y='F std (eV/atom)',
                        markers=True,
                        line_dash_sequence=['dot'],
                    )
                    fig.update_layout(
                        xaxis_title=f'{dopant_B} (at.%)',
                        yaxis_title='Forming Energy (eV/atom)',
                        yaxis_tickformat='.2f',
                    )
                    st.plotly_chart(fig, width='stretch')
                
                with col6:
                    # Plot the F (eV/atom) vs A-site dopant concentration
                    if dopant_B_conc[0] == dopant_B_conc[1]:
                        dopant_B_conc_filter = dopant_B_conc[0]
                    else:
                        dopant_B_conc_filter = st.slider(
                            label=f'Adjust the Concentration of {dopant_B} (at.%):',
                            min_value=dopant_B_conc[0],
                            max_value=dopant_B_conc[1],
                            value=(dopant_B_conc[0] + dopant_B_conc[1]) // 2,
                            step=1,
                            key='dopant_B_conc_filter_for_F_B'
                        )
                    data_dis_filter = data_dis[data_dis['Dopant Concentration B (at.%)'] == dopant_B_conc_filter]
                    fig = px.line(
                        data_dis_filter,
                        x='Dopant Concentration A (at.%)',
                        y='F (eV/atom)',
                        error_y='F std (eV/atom)',
                        markers=True,
                        line_dash_sequence=['dot'],
                    )
                    fig.update_layout(
                        xaxis_title=f'{dopant_A} (at.%)',
                        yaxis_title='Forming Energy (eV/atom)',
                        yaxis_tickformat='.2f',
                    )
                    st.plotly_chart(fig, width='stretch')
            
            elif dopant_A != 'None' and dopant_B == 'None':
                fig = px.line(
                    data_dis,
                    x='Dopant Concentration A (at.%)',
                    y='F (eV/atom)',
                    error_y='F std (eV/atom)',
                    markers=True,
                    line_dash_sequence=['dot'],
                )
                fig.update_layout(
                    xaxis_title=f'{dopant_A} (at.%)',
                    yaxis_title='Forming Energy (eV/atom)',
                    yaxis_tickformat='.2f',
                )
                st.plotly_chart(fig, width='stretch')
            
            elif dopant_A == 'None' and dopant_B != 'None':
                fig = px.line(
                    data_dis,
                    x='Dopant Concentration B (at.%)',
                    y='F (eV/atom)',
                    error_y='F std (eV/atom)',
                    markers=True,
                    line_dash_sequence=['dot'],
                )
                fig.update_layout(
                    xaxis_title=f'{dopant_B} (at.%)',
                    yaxis_title='Forming Energy (eV/atom)',
                    yaxis_tickformat='.2f',
                )
                st.plotly_chart(fig, width='stretch')

        # Visualize the lattice distortion and atomic distortion
        st.divider()
        st.subheader('Structural Distortions:')
        col7, col8 = st.columns(2, border=True, gap='medium')
        with col7:
            st.markdown('**Lattice Distortion (%):**')
            # Create a heatmap for lattice distortion
            if dopant_A != 'None' and dopant_B != 'None':
                heatmap_lattice = data_dis.pivot_table(index='Dopant Concentration A (at.%)', columns='Dopant Concentration B (at.%)', values='Lattice Distortion (%)')
                labels=dict(x=f'{dopant_B} (at.%)', y=f'{dopant_A} (at.%)', color='Lattice Distortion (%)')
            elif dopant_A != 'None' and dopant_B == 'None':
                heatmap_lattice = data_dis.pivot_table(index='Dopant Concentration B (at.%)', columns='Dopant Concentration A (at.%)', values='Lattice Distortion (%)')
                labels=dict(x=f'{dopant_A} (at.%)', y='', color='Lattice Distortion (%)')
            elif dopant_A == 'None' and dopant_B != 'None':
                heatmap_lattice = data_dis.pivot_table(index='Dopant Concentration A (at.%)', columns='Dopant Concentration B (at.%)', values='Lattice Distortion (%)')
                labels=dict(x=f'{dopant_B} (at.%)', y='', color='Lattice Distortion (%)')
            # Create the Plotly heatmap
            fig = px.imshow(
                heatmap_lattice,
                color_continuous_scale='Viridis',
                labels=labels,
            )
            fig.update_coloraxes(
                colorbar_tickformat='.2f',
                colorbar_title_text='',
            )
            # Display in Streamlit
            st.plotly_chart(fig, width='stretch')
        
        with col8:
            st.markdown('**Atomic Distortion (Å):**')
            # Create a heatmap for atomic distortion
            if dopant_A != 'None' and dopant_B != 'None':
                heatmap_atomic = data_dis.pivot_table(index='Dopant Concentration A (at.%)', columns='Dopant Concentration B (at.%)', values='Atomic Distortion (Å)')
                labels=dict(x=f'{dopant_B} (at.%)', y=f'{dopant_A} (at.%)', color='Atomic Distortion (Å)')
            elif dopant_A != 'None' and dopant_B == 'None':
                heatmap_atomic = data_dis.pivot_table(index='Dopant Concentration B (at.%)', columns='Dopant Concentration A (at.%)', values='Atomic Distortion (Å)')
                labels=dict(x=f'{dopant_A} (at.%)', y='', color='Atomic Distortion (Å)')
            elif dopant_A == 'None' and dopant_B != 'None':
                heatmap_atomic = data_dis.pivot_table(index='Dopant Concentration A (at.%)', columns='Dopant Concentration B (at.%)', values='Atomic Distortion (Å)')
                labels=dict(x=f'{dopant_B} (at.%)', y='', color='Atomic Distortion (Å)')
            # Create the Plotly heatmap
            fig = px.imshow(
                heatmap_atomic,
                color_continuous_scale='Viridis',
                labels=labels,
            )
            fig.update_coloraxes(
                colorbar_tickformat='.2f',
                colorbar_title_text='',
            )
            # Display in Streamlit
            st.plotly_chart(fig, width='stretch')
        
        with st.expander('**Lattice Distortion Trends Across Different Dopant Sites** (Click to Expand)', expanded=False):
            if dopant_A != 'None' and dopant_B != 'None':
                # Set 2 columns for A and B site dopant
                col9, col10 = st.columns(2, border=True, gap='medium')
                with col9:
                    # Plot the lattice distortion vs B-site dopant concentration
                    if dopant_A_conc[0] == dopant_A_conc[1]:
                        dopant_A_conc_filter = dopant_A_conc[0]
                    else:
                        dopant_A_conc_filter = st.slider(
                            label=f'Adjust the Concentration of {dopant_A} (at.%):',
                            min_value=dopant_A_conc[0],
                            max_value=dopant_A_conc[1],
                            value=(dopant_A_conc[0] + dopant_A_conc[1]) // 2,
                            step=1,
                            key='dopant_A_conc_filter_for_lattice_A'
                        )
                    data_dis_filter = data_dis[data_dis['Dopant Concentration A (at.%)'] == dopant_A_conc_filter]
                    fig = px.line(
                        data_dis_filter,
                        x='Dopant Concentration B (at.%)',
                        y='Lattice Distortion (%)',
                        error_y='Lattice Distortion std (%)',
                        markers=True,
                        line_dash_sequence=['dot'],
                    )
                    fig.update_layout(
                        xaxis_title=f'{dopant_B} (at.%)',
                        yaxis_title='Lattice Distortion (%)',
                        yaxis_tickformat='.2f',
                    )
                    st.plotly_chart(fig, width='stretch')
                
                with col10:
                    # Plot the lattice distortion vs A-site dopant concentration
                    if dopant_B_conc[0] == dopant_B_conc[1]:
                        dopant_B_conc_filter = dopant_B_conc[0]
                    else:
                        dopant_B_conc_filter = st.slider(
                            label=f'Adjust the Concentration of {dopant_B} (at.%):',
                            min_value=dopant_B_conc[0],
                            max_value=dopant_B_conc[1],
                            value=(dopant_B_conc[0] + dopant_B_conc[1]) // 2,
                            step=1,
                            key='dopant_B_conc_filter_for_lattice_B'
                        )
                    data_dis_filter = data_dis[data_dis['Dopant Concentration B (at.%)'] == dopant_B_conc_filter]
                    fig = px.line(
                        data_dis_filter,
                        x='Dopant Concentration A (at.%)',
                        y='Lattice Distortion (%)',
                        error_y='Lattice Distortion std (%)',
                        markers=True,
                        line_dash_sequence=['dot'],
                    )
                    fig.update_layout(
                        xaxis_title=f'{dopant_A} (at.%)',
                        yaxis_title='Lattice Distortion (%)',
                        yaxis_tickformat='.2f',
                    )
                    st.plotly_chart(fig, width='stretch')
            
            elif dopant_A != 'None' and dopant_B == 'None':
                fig = px.line(
                    data_dis,
                    x='Dopant Concentration A (at.%)',
                    y='Lattice Distortion (%)',
                    error_y='Lattice Distortion std (%)',
                    markers=True,
                    line_dash_sequence=['dot'],
                )
                fig.update_layout(
                    xaxis_title=f'{dopant_A} (at.%)',
                    yaxis_title='Lattice Distortion (%)',
                    yaxis_tickformat='.2f',
                )
                st.plotly_chart(fig, width='stretch')
            
            elif dopant_A == 'None' and dopant_B != 'None':
                fig = px.line(
                    data_dis,
                    x='Dopant Concentration B (at.%)',
                    y='Lattice Distortion (%)',
                    error_y='Lattice Distortion std (%)',
                    markers=True,
                    line_dash_sequence=['dot'],
                )
                fig.update_layout(
                    xaxis_title=f'{dopant_B} (at.%)',
                    yaxis_title='Lattice Distortion (%)',
                    yaxis_tickformat='.2f',
                )
                st.plotly_chart(fig, width='stretch')

        with st.expander('**Atomic Distortion Trends Across Different Dopant Sites** (Click to Expand)', expanded=False):
            if dopant_A != 'None' and dopant_B != 'None':
                # Set 2 columns for A and B site dopant
                col11, col12 = st.columns(2, border=True, gap='medium')
                with col11:
                    # Plot the atomic distortion vs B-site dopant concentration
                    if dopant_A_conc[0] == dopant_A_conc[1]:
                        dopant_A_conc_filter = dopant_A_conc[0]
                    else:   
                        dopant_A_conc_filter = st.slider(
                            label=f'Adjust the Concentration of {dopant_A} (at.%):',
                            min_value=dopant_A_conc[0],
                            max_value=dopant_A_conc[1],
                            value=(dopant_A_conc[0] + dopant_A_conc[1]) // 2,
                            step=1,
                            key='dopant_A_conc_filter_for_atomic_A'
                        )
                    data_dis_filter = data_dis[data_dis['Dopant Concentration A (at.%)'] == dopant_A_conc_filter]
                    fig = px.line(
                        data_dis_filter,
                        x='Dopant Concentration B (at.%)',
                        y='Atomic Distortion (Å)',
                        error_y='Atomic Distortion std (Å)',
                        markers=True,
                        line_dash_sequence=['dot'],
                    )
                    fig.update_layout(
                        xaxis_title=f'{dopant_B} (at.%)',
                        yaxis_title='Atomic Distortion (Å)',
                        yaxis_tickformat='.2f',
                    )
                    st.plotly_chart(fig, width='stretch')
                
                with col12:
                    # Plot the atomic distortion vs A-site dopant concentration
                    if dopant_B_conc[0] == dopant_B_conc[1]:
                        dopant_B_conc_filter = dopant_B_conc[0]
                    else:
                        dopant_B_conc_filter = st.slider(
                            label=f'Adjust the Concentration of {dopant_B} (at.%):',
                            min_value=dopant_B_conc[0],
                            max_value=dopant_B_conc[1],
                            value=(dopant_B_conc[0] + dopant_B_conc[1]) // 2,
                            step=1,
                            key='dopant_B_conc_filter_for_atomic_B'
                        )
                    data_dis_filter = data_dis[data_dis['Dopant Concentration B (at.%)'] == dopant_B_conc_filter]
                    fig = px.line(
                        data_dis_filter,
                        x='Dopant Concentration A (at.%)',
                        y='Atomic Distortion (Å)',
                        error_y='Atomic Distortion std (Å)',
                        markers=True,
                        line_dash_sequence=['dot'],
                    )
                    fig.update_layout(
                        xaxis_title=f'{dopant_A} (at.%)',
                        yaxis_title='Atomic Distortion (Å)',
                        yaxis_tickformat='.2f',
                    )
                    st.plotly_chart(fig, width='stretch')

            elif dopant_A != 'None' and dopant_B == 'None':
                fig = px.line(
                    data_dis,
                    x='Dopant Concentration A (at.%)',
                    y='Atomic Distortion (Å)',
                    error_y='Atomic Distortion std (Å)',
                    markers=True,
                    line_dash_sequence=['dot'],
                )
                fig.update_layout(
                    xaxis_title=f'{dopant_A} (at.%)',
                    yaxis_title='Atomic Distortion (Å)',
                    yaxis_tickformat='.2f',
                )
                st.plotly_chart(fig, width='stretch')

            elif dopant_A == 'None' and dopant_B != 'None':
                fig = px.line(
                    data_dis,
                    x='Dopant Concentration B (at.%)',
                    y='Atomic Distortion (Å)',
                    error_y='Atomic Distortion std (Å)',
                    markers=True,
                    line_dash_sequence=['dot'],
                )
                fig.update_layout(
                    xaxis_title=f'{dopant_B} (at.%)',
                    yaxis_title='Atomic Distortion (Å)',
                    yaxis_tickformat='.2f',
                )
                st.plotly_chart(fig, width='stretch')

        # Visualize the diffusion coefficient
        st.divider()
        st.subheader('Diffusivity & Conductivity:')
        col13, col14 = st.columns(2, border=True, gap='medium')
        with col13:
            st.markdown('**Diffusion Coefficient (cm<sup>2</sup>/s):**', unsafe_allow_html=True)
            # Create a heatmap for D (cm^2/s)
            if dopant_A != 'None' and dopant_B != 'None':
                heatmap_D = data_dis.pivot_table(index='Dopant Concentration A (at.%)', columns='Dopant Concentration B (at.%)', values='D (cm^2/s)')
                labels=dict(x=f'{dopant_B} (at.%)', y=f'{dopant_A} (at.%)', color='D (cm^2/s)')
            elif dopant_A != 'None' and dopant_B == 'None':
                heatmap_D = data_dis.pivot_table(index='Dopant Concentration B (at.%)', columns='Dopant Concentration A (at.%)', values='D (cm^2/s)')
                labels=dict(x=f'{dopant_A} (at.%)', y='', color='D (cm^2/s)')
            elif dopant_A == 'None' and dopant_B != 'None':
                heatmap_D = data_dis.pivot_table(index='Dopant Concentration A (at.%)', columns='Dopant Concentration B (at.%)', values='D (cm^2/s)')
                labels=dict(x=f'{dopant_B} (at.%)', y='', color='D (cm^2/s)')
            # Create the Plotly heatmap
            fig = px.imshow(
                heatmap_D,
                color_continuous_scale='Viridis',
                labels=labels,
            )
            fig.update_coloraxes(
                colorbar_tickformat='.2e',
                colorbar_title_text='',
            )
            # Display in Streamlit
            st.plotly_chart(fig, width='stretch')
        with col14:
            st.markdown('**Ionic Conductivity (S/cm):**')
            # Create a heatmap for C (S/cm)
            if dopant_A != 'None' and dopant_B != 'None':
                heatmap_C = data_dis.pivot_table(index='Dopant Concentration A (at.%)', columns='Dopant Concentration B (at.%)', values='C (S/cm)')
                labels=dict(x=f'{dopant_B} (at.%)', y=f'{dopant_A} (at.%)', color='C (S/cm)')
            elif dopant_A != 'None' and dopant_B == 'None':
                heatmap_C = data_dis.pivot_table(index='Dopant Concentration B (at.%)', columns='Dopant Concentration A (at.%)', values='C (S/cm)')
                labels=dict(x=f'{dopant_A} (at.%)', y='', color='C (S/cm)')
            elif dopant_A == 'None' and dopant_B != 'None':
                heatmap_C = data_dis.pivot_table(index='Dopant Concentration A (at.%)', columns='Dopant Concentration B (at.%)', values='C (S/cm)')
                labels=dict(x=f'{dopant_B} (at.%)', y='', color='C (S/cm)')
            # Create the Plotly heatmap
            fig = px.imshow(
                heatmap_C,
                color_continuous_scale='Viridis',
                labels=labels,
            )
            fig.update_coloraxes(
                colorbar_tickformat='.2e',
                colorbar_title_text='',
            )
            # Display in Streamlit
            st.plotly_chart(fig, width='stretch')
        
        with st.expander('**Diffusion Coefficient Trends Across Different Dopant Sites** (Click to Expand)', expanded=False):
            if dopant_A != 'None' and dopant_B != 'None':
                # Set 2 columns for A and B site dopant
                col15, col16 = st.columns(2, border=True, gap='medium')
                with col15:
                    # Plot the D (cm^2/s) vs B-site dopant concentration
                    if dopant_A_conc[0] == dopant_A_conc[1]:
                        dopant_A_conc_filter = dopant_A_conc[0]
                    else:
                        dopant_A_conc_filter = st.slider(
                            label=f'Adjust the Concentration of {dopant_A} (at.%):',
                            min_value=dopant_A_conc[0],
                            max_value=dopant_A_conc[1],
                            value=(dopant_A_conc[0] + dopant_A_conc[1]) // 2,
                            step=1,
                            key='dopant_A_conc_filter_for_D_A'
                        )
                    data_dis_filter = data_dis[data_dis['Dopant Concentration A (at.%)'] == dopant_A_conc_filter]
                    fig = px.line(
                        data_dis_filter,
                        x='Dopant Concentration B (at.%)',
                        y='D (cm^2/s)',
                        markers=True,
                        line_dash_sequence=['dot'],
                    )
                    fig.update_layout(
                        xaxis_title=f'{dopant_B} (at.%)',
                        yaxis_title='Diffusion Coefficient (cm<sup>2</sup>/s)',
                        yaxis_tickformat='.2e',
                    )
                    st.plotly_chart(fig, width='stretch')
                
                with col16:
                    # Plot the D (cm^2/s) vs A-site dopant concentration
                    if dopant_B_conc[0] == dopant_B_conc[1]:
                        dopant_B_conc_filter = dopant_B_conc[0]
                    else:
                        dopant_B_conc_filter = st.slider(
                            label=f'Adjust the Concentration of {dopant_B} (at.%):',
                            min_value=dopant_B_conc[0],
                            max_value=dopant_B_conc[1],
                            value=(dopant_B_conc[0] + dopant_B_conc[1]) // 2,
                            step=1,
                            key='dopant_B_conc_filter_for_D_B'
                        )
                    data_dis_filter = data_dis[data_dis['Dopant Concentration B (at.%)'] == dopant_B_conc_filter]
                    fig = px.line(
                        data_dis_filter,
                        x='Dopant Concentration A (at.%)',
                        y='D (cm^2/s)',
                        markers=True,
                        line_dash_sequence=['dot'],
                    )
                    fig.update_layout(
                        xaxis_title=f'{dopant_A} (at.%)',
                        yaxis_title='Diffusion Coefficient (cm<sup>2</sup>/s)',
                        yaxis_tickformat='.2e',
                    )
                    st.plotly_chart(fig, width='stretch')
            
            elif dopant_A != 'None' and dopant_B == 'None':
                fig = px.line(
                    data_dis,
                    x='Dopant Concentration A (at.%)',
                    y='D (cm^2/s)',
                    markers=True,
                    line_dash_sequence=['dot'],
                )
                fig.update_layout(
                    xaxis_title=f'{dopant_A} (at.%)',
                    yaxis_title='Diffusion Coefficient (cm<sup>2</sup>/s)',
                    yaxis_tickformat='.2e',
                )
                st.plotly_chart(fig, width='stretch')
            
            elif dopant_A == 'None' and dopant_B != 'None':
                fig = px.line(
                    data_dis,
                    x='Dopant Concentration B (at.%)',
                    y='D (cm^2/s)',
                    markers=True,
                    line_dash_sequence=['dot'],
                )
                fig.update_layout(
                    xaxis_title=f'{dopant_B} (at.%)',
                    yaxis_title='Diffusion Coefficient (cm<sup>2</sup>/s)',
                    yaxis_tickformat='.2e',
                )
                st.plotly_chart(fig, width='stretch')

        # Visualize the ionic conductivity  
        with st.expander('**Ionic Conductivity Trends Across Different Dopant Sites** (Click to Expand)', expanded=False):
            if dopant_A != 'None' and dopant_B != 'None':
                # Set 2 columns for A and B site dopant
                col17, col18 = st.columns(2, border=True, gap='medium')
                with col17:
                    # Plot the C (S/cm) vs B-site dopant concentration
                    if dopant_A_conc[0] == dopant_A_conc[1]:
                        dopant_A_conc_filter = dopant_A_conc[0]
                    else:
                        dopant_A_conc_filter = st.slider(
                            label=f'Adjust the Concentration of {dopant_A} (at.%):',
                            min_value=dopant_A_conc[0],
                            max_value=dopant_A_conc[1],
                            value=(dopant_A_conc[0] + dopant_A_conc[1]) // 2,
                            step=1,
                            key='dopant_A_conc_filter_for_C_A'
                        )
                    data_dis_filter = data_dis[data_dis['Dopant Concentration A (at.%)'] == dopant_A_conc_filter]
                    fig = px.line(
                        data_dis_filter,
                        x='Dopant Concentration B (at.%)',
                        y='C (S/cm)',
                        markers=True,
                        line_dash_sequence=['dot'],
                    )
                    fig.update_layout(
                        xaxis_title=f'{dopant_B} (at.%)',
                        yaxis_title='Ionic Conductivity (S/cm)',
                        yaxis_tickformat='.2e',
                    )
                    st.plotly_chart(fig, width='stretch')
                
                with col18:
                    # Plot the C (S/cm) vs A-site dopant concentration
                    if dopant_B_conc[0] == dopant_B_conc[1]:
                        dopant_B_conc_filter = dopant_B_conc[0]
                    else:
                        dopant_B_conc_filter = st.slider(
                            label=f'Adjust the Concentration of {dopant_B} (at.%):',
                            min_value=dopant_B_conc[0],
                            max_value=dopant_B_conc[1],
                            value=(dopant_B_conc[0] + dopant_B_conc[1]) // 2,
                            step=1,
                            key='dopant_B_conc_filter_for_C_B'
                        )
                    data_dis_filter = data_dis[data_dis['Dopant Concentration B (at.%)'] == dopant_B_conc_filter]
                    fig = px.line(
                        data_dis_filter,
                        x='Dopant Concentration A (at.%)',
                        y='C (S/cm)',
                        markers=True,
                        line_dash_sequence=['dot'],
                    )
                    fig.update_layout(
                        xaxis_title=f'{dopant_A} (at.%)',
                        yaxis_title='Ionic Conductivity (S/cm)',
                        yaxis_tickformat='.2e',
                    )
                    st.plotly_chart(fig, width='stretch')
            
            elif dopant_A != 'None' and dopant_B == 'None':
                fig = px.line(
                    data_dis,
                    x='Dopant Concentration A (at.%)',
                    y='C (S/cm)',
                    markers=True,
                    line_dash_sequence=['dot'],
                )
                fig.update_layout(
                    xaxis_title=f'{dopant_A} (at.%)',
                    yaxis_title='Ionic Conductivity (S/cm)',
                    yaxis_tickformat='.2e',
                )
                st.plotly_chart(fig, width='stretch')

            elif dopant_A == 'None' and dopant_B != 'None':
                fig = px.line(
                    data_dis,
                    x='Dopant Concentration B (at.%)',
                    y='C (S/cm)',
                    markers=True,
                    line_dash_sequence=['dot'],
                )
                fig.update_layout(
                    xaxis_title=f'{dopant_B} (at.%)',
                    yaxis_title='Ionic Conductivity (S/cm)',
                    yaxis_tickformat='.2e',
                )
                st.plotly_chart(fig, width='stretch')

    # Add a footer
    st.divider()
    st.markdown(
    '''
    <div style='color: rgba(0, 0, 0, 0.4); font-weight: bold;'>
        Copyright © 2025 HEP-Explorer | Developed by Guangchen Liu. All rights reserved.
    </div>
    ''',
    unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()