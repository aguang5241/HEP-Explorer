import json
import hashlib
import pandas as pd
import numpy as np
import streamlit as st
import torch
import pyro
import joblib
import plotly.express as px
import plotly.graph_objects as go


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


def BNN_predictor(site, data_pre):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the x and y scalers and the train data
    if site == 'A-site':
        with open('res/x_scaler_A.pkl', 'rb') as f:
            x_scaler = joblib.load(f)
        with open('res/y_scaler_A.pkl', 'rb') as f:
            y_scaler = joblib.load(f)
        with open('res/x_train_A.pkl', 'rb') as f:
            x_train = joblib.load(f)
        with open('res/y_train_A.pkl', 'rb') as f:
            y_train = joblib.load(f)
        model_path = 'res/model_A.pkl'
    else:
        with open('res/x_scaler_B.pkl', 'rb') as f:
            x_scaler = joblib.load(f)
        with open('res/y_scaler_B.pkl', 'rb') as f:
            y_scaler = joblib.load(f)
        with open('res/x_train_B.pkl', 'rb') as f:
            x_train = joblib.load(f)
        with open('res/y_train_B.pkl', 'rb') as f:
            y_train = joblib.load(f)
        model_path = 'res/model_B.pkl'

    # Get the training data
    train_x = torch.FloatTensor(x_train).to(DEVICE)
    train_y = torch.FloatTensor(y_train).to(DEVICE)

    # Get the input data to predict
    x_pre = data_pre[[
        'atomic_number_1', 'atomic_number_2', 'atomic_number_3', 'atomic_number_4',
        'ionic_radius_1', 'ionic_radius_2', 'ionic_radius_3', 'ionic_radius_4',
        'atomic_concentration_1', 'atomic_concentration_2', 'atomic_concentration_3', 'atomic_concentration_4',
        'T'
    ]].to_numpy()
    x_pre = x_scaler.transform(x_pre)
    x_pre = torch.FloatTensor(x_pre).to(DEVICE)

    # Define the model
    if site == 'A-site':
        model_path = 'res/model_A.pkl'
        IN_FEATURES = 13
        OUT_FEATURES = 4
        N_LAYERS = 2
        PRIOR_SCALE_WEIGHT = 0.4082109636101447
        PRIOR_SCALE_BIAS = 0.5321222164940764
        NODES_LIST = [26, 34]
        LR = 0.009981166029931921
    else:
        model_path = 'res/model_B.pkl'
        IN_FEATURES = 13
        OUT_FEATURES = 4
        N_LAYERS = 2
        PRIOR_SCALE_WEIGHT = 0.5552777841477516
        PRIOR_SCALE_BIAS = 0.03737628946174844
        NODES_LIST = [26, 28]
        LR = 0.009940260795305739

    # Start the prediction
    PREDICTIVE_SAMPLES = 1000
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
    predictive = pyro.infer.Predictive(model, guide=guide, num_samples=PREDICTIVE_SAMPLES)
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
        data_pred = pd.concat([data_pre, pd.DataFrame(y_pre, columns=['Ef', 'D_lattice', 'D_atomic', 'D_negative']), pd.DataFrame(y_std, columns=['Ef_std', 'D_lattice_std', 'D_atomic_std', 'D_negative_std'])], axis=1)

    return data_pred


def get_dopant_data(site, element_conc, temperature):
    # Load the compositional space
    compositional_space = joblib.load('res/data_all.pkl')

    # Filter the compositional space based on the selected elements and their concentration ranges
    elements = list(element_conc.keys())
    atom_ranges = [(int(conc[0]/100*80), int(conc[1]/100*80)) for conc in element_conc.values()]
    a_ranges, b_ranges, c_ranges, d_ranges, e_ranges = atom_ranges
    data_pre = compositional_space.copy()
    data_pre = data_pre[
        (data_pre['a'] >= a_ranges[0]) & (data_pre['a'] <= a_ranges[1]) &
        (data_pre['b'] >= b_ranges[0]) & (data_pre['b'] <= b_ranges[1]) &
        (data_pre['c'] >= c_ranges[0]) & (data_pre['c'] <= c_ranges[1]) &
        (data_pre['d'] >= d_ranges[0]) & (data_pre['d'] <= d_ranges[1]) &
        (data_pre['e'] >= e_ranges[0]) & (data_pre['e'] <= e_ranges[1])
    ].reset_index(drop=True)
    data_pre = data_pre.rename(columns={'a': elements[0], 'b': elements[1], 'c': elements[2], 'd': elements[3], 'e': elements[4]})

    # Remove the column of La or Co
    if site == 'A-site':
        data_pre = data_pre.drop(columns=['La'])
    else:
        data_pre = data_pre.drop(columns=['Co'])

    # Get the elements from the columns
    elements_ = data_pre.columns.tolist()

    # Define the atomic numbers and ionic radii for the substitutional elements
    atomic_numbers = {
        'Ca': 20, 'Sr': 38, 'Ba': 56, 'Ce': 58, 
        'Pr': 59, 'Nd': 60, 'Sm': 62, 'Gd': 64, 
        'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 
        'Fe': 26, 'Ni': 28, 'Cu': 29, 'Zn': 30
        }
    ionic_radii = {
        'Ca': 1.340, 'Sr': 1.440, 'Ba': 1.610, 'Ce': 1.340, 
        'Pr': 1.179, 'Nd': 1.270, 'Sm': 1.240, 'Gd': 1.107, 
        'Ti': 0.670, 'V': 0.640, 'Cr': 0.615, 'Mn': 0.645, 
        'Fe': 0.645, 'Ni': 0.600, 'Cu': 0.540, 'Zn': 0.740
        }
    
    # Get the atomic number and ionic radius based on elements
    atomic_numbers_list = [atomic_numbers[elem] for elem in elements_]
    ionic_radii_list = [ionic_radii[elem] for elem in elements_]
    data_pre['atomic_number_1'] = atomic_numbers_list[0]
    data_pre['atomic_number_2'] = atomic_numbers_list[1]
    data_pre['atomic_number_3'] = atomic_numbers_list[2]
    data_pre['atomic_number_4'] = atomic_numbers_list[3]
    data_pre['ionic_radius_1'] = ionic_radii_list[0]
    data_pre['ionic_radius_2'] = ionic_radii_list[1]
    data_pre['ionic_radius_3'] = ionic_radii_list[2]
    data_pre['ionic_radius_4'] = ionic_radii_list[3]
    data_pre['atomic_concentration_1'] = data_pre[elements_[0]]/0.80
    data_pre['atomic_concentration_2'] = data_pre[elements_[1]]/0.80
    data_pre['atomic_concentration_3'] = data_pre[elements_[2]]/0.80
    data_pre['atomic_concentration_4'] = data_pre[elements_[3]]/0.80
    data_pre['T'] = temperature

    return data_pre


def D2C(D, T):
    z = 2                           # charge number of the Oxygen ion
    F = 9.64853399e4                # Faraday constant, C/mol
    R = 8.31446261815324            # J/(K・mol)
    C = z**2 * D * F**2 / (R*T*100) # S/cm
    return C


def data_display(data_pred):
    data_dis = pd.DataFrame()
    data_dis['concentration_1 (at.%)'] = data_pred['atomic_concentration_1']
    data_dis['concentration_2 (at.%)'] = data_pred['atomic_concentration_2']
    data_dis['concentration_3 (at.%)'] = data_pred['atomic_concentration_3']
    data_dis['concentration_4 (at.%)'] = data_pred['atomic_concentration_4']
    data_dis['b'] = (data_pred['atomic_concentration_1'] * 0.80).round().astype(int)
    data_dis['c'] = (data_pred['atomic_concentration_2'] * 0.80).round().astype(int)
    data_dis['d'] = (data_pred['atomic_concentration_3'] * 0.80).round().astype(int)
    data_dis['e'] = (data_pred['atomic_concentration_4'] * 0.80).round().astype(int)
    data_dis['a'] = 80 - data_dis['b'] - data_dis['c'] - data_dis['d'] - data_dis['e']
    data_dis['Temperature (K)'] = data_pred['T'].astype(int)
    data_dis['Ef (eV/atom)'] = data_pred['Ef']
    data_dis['D_lattice (cm^2/s)'] = data_pred['D_lattice']
    data_dis['D_atomic (cm^2/s)'] = data_pred['D_atomic']
    data_dis['D (cm^2/s)'] = -data_pred['D_negative']
    data_dis['Ef std (eV/atom)'] = data_pred['Ef_std']
    data_dis['D_lattice std (cm^2/s)'] = data_pred['D_lattice_std']
    data_dis['D_atomic std (cm^2/s)'] = data_pred['D_atomic_std']
    data_dis['D std (cm^2/s)'] = data_pred['D_negative_std']
    data_dis['C (S/cm)'] = D2C(data_dis['D (cm^2/s)'], data_dis['Temperature (K)'])
    data_dis['element_1'] = data_pred.columns[0]
    data_dis['element_2'] = data_pred.columns[1]
    data_dis['element_3'] = data_pred.columns[2]
    data_dis['element_4'] = data_pred.columns[3]
    data_dis = data_dis[[
        'element_1', 'concentration_1 (at.%)',
        'element_2', 'concentration_2 (at.%)', 
        'element_3', 'concentration_3 (at.%)', 
        'element_4', 'concentration_4 (at.%)', 
        'Temperature (K)', 
        'Ef (eV/atom)', 'D_lattice (cm^2/s)', 'D_atomic (cm^2/s)', 'D (cm^2/s)', 
        'Ef std (eV/atom)', 'D_lattice std (cm^2/s)', 'D_atomic std (cm^2/s)', 'D std (cm^2/s)', 
        'C (S/cm)',
        'a', 'b', 'c', 'd', 'e'
        ]]
    return data_dis


def set_labels_plotly(fig, labels):
    fig.add_annotation(x=0.82, y=0.94, text=labels[0],
                       xref="paper", yref="paper",
                       showarrow=False, font=dict(size=12, color="black"))
    fig.add_annotation(x=0.04, y=0.84, text=labels[1],
                       xref="paper", yref="paper",
                       showarrow=False, font=dict(size=12, color="black"))
    fig.add_annotation(x=0.92, y=0.35, text=labels[2],
                       xref="paper", yref="paper",
                       showarrow=False, font=dict(size=12, color="black"))
    fig.add_annotation(x=0.02, y=0.20, text=labels[3],
                       xref="paper", yref="paper",
                       showarrow=False, font=dict(size=12, color="black"))
    fig.add_annotation(x=0.54, y=0.02, text=labels[4],
                       xref="paper", yref="paper",
                       showarrow=False, font=dict(size=12, color="black"))
    return fig


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
        elements = st.multiselect(
            label='Please select five A-site element(s) for substitution:',
            options=element_pool,
            default=['La', 'Ca', 'Sr', 'Ba', 'Ce'],
            max_selections=5,
        )
        # Ensure La is always included
        if 'La' not in elements:
            st.error("Error: La is mandatory for A-site substitution.")
        # Ensure 5 elements are selected
        if len(elements) != 5:
            st.error("Error: Please select exactly five elements.")
    else:
        element_pool = ['Co', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Cu', 'Zn']
        elements = st.multiselect(
            label='Please select five B-site element(s) for substitution:',
            options=element_pool,
            default=['Co', 'Ni', 'Fe', 'Mn', 'Cr'],
            max_selections=5,
        )
        # Ensure Co is always included
        if 'Co' not in elements:
            st.error("Error: Co is mandatory for B-site substitution.")
        # Ensure 5 elements are selected
        if len(elements) != 5:
            st.error("Error: Please select exactly five elements.")

    # Add sliders for dopant concentration range selection
    st.subheader('Substitutional Concentrations:')
    st.caption('Please select the concentration range (15-40 at.%) for each selected element.')
    concentration_container = st.container(border=True, gap='medium', width='stretch')
    element_conc = {}
    with concentration_container:
        if select_site == 'A-site':
            for i, elem in enumerate(elements):
                conc = st.slider(
                    label=f'**{elem}** (at.%):',
                    min_value=15,
                    max_value=40,
                    value=(15, 25),
                    step=1,
                    key=f'slider_A_{i}',
                )
                element_conc[elem] = conc
        else:
            for i, elem in enumerate(elements):
                conc = st.slider(
                    label=f'**{elem}** (at.%):',
                    min_value=15,
                    max_value=40,
                    value=(15, 25),
                    step=1,
                    key=f'slider_B_{i}',
                )
                element_conc[elem] = conc

    # Add a container for system conditions
    st.subheader('System Conditions:')
    st.caption('This only applies to diffusivity and conductivity predictions.')
    temperature_container = st.container(border=True, gap='medium', width='stretch')

    # Set slider for temperature
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
        "element_conc": element_conc,
        "temperature": T,
    }, sort_keys=True)

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
            if 'La' not in elements:
                st.error("Error: La is mandatory for A-site substitution.")
                st.stop()
        else:
            if 'Co' not in elements:
                st.error("Error: Co is mandatory for B-site substitution.")
                st.stop()
        # Ensure 5 elements are selected
        if len(elements) != 5:
            st.error("Error: Please select exactly five elements.")
            st.stop()

        # Show a loading spinner
        with st.spinner('Predicting... Please wait — this may take a few seconds to a few minutes depending on your input size and complexity.'):
            try:
                # Get the dopant data
                data_pre = get_dopant_data(site=select_site, element_conc=element_conc, temperature=T)
                # Predict the data using BNN
                data_pred = BNN_predictor(site=select_site, data_pre=data_pre)
                # Process the predicted data for display
                st.session_state['data_pred_dis'] = data_display(data_pred=data_pred)
            except Exception as e:
                # Clear previous predictions
                st.session_state['data_pred_dis'] = None
                # Show user-friendly error message
                st.error('😢 Sorry, something unexpected happened. Please try again.')
                st.stop()
    
    # Show the predicted data
    if st.session_state.get('data_pred_dis') is not None:
        data_dis = st.session_state['data_pred_dis']
        # Filter the data to show only rows with D > 0
        data_dis = data_dis[data_dis['D (cm^2/s)'] > 0].reset_index(drop=True)
        umap_model = joblib.load('res/umap_model.pkl')
        data_all = joblib.load('res/data_all.pkl')
        umap_embeddings = umap_model.transform(data_all[['a', 'b', 'c', 'd', 'e']].values)
        data_dis_embeddings = umap_model.transform(data_dis[['a', 'b', 'c', 'd', 'e']].values)
        labels = ['La' if select_site == 'A-site' else 'Co', data_dis['element_1'].iloc[0], data_dis['element_2'].iloc[0], data_dis['element_3'].iloc[0], data_dis['element_4'].iloc[0]]

        # Visualize the forming energy, lattice distortion, atomic distortion, and diffusivity
        st.divider()
        col1, col2 = st.columns(2, gap='medium', border=True)
        col3, col4 = st.columns(2, gap='medium', border=True)
        # col5, col6 = st.columns(2, gap='medium', border=True)
        kwargs_data_all = dict(
            x=umap_embeddings[:, 0],
            y=umap_embeddings[:, 1],
            mode="markers",
            marker=dict(color="lightgrey", size=5, symbol="circle"),
            name="Compositional Space",
            opacity=0.5,
        )
        kwargs_data_dis = dict(
            x=data_dis_embeddings[:, 0],
            y=data_dis_embeddings[:, 1],
            mode='markers',
            name='Predicted Compositions',
        )
        kwargs_fig = dict(
            # width=600,
            # height=600,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=-0.2,
                xanchor='center',
                x=0.5
            )
        )

        # Formation Energy
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(**kwargs_data_all))
            fig.add_trace(
                go.Scatter(
                    **kwargs_data_dis,
                    marker=dict(
                        color=data_dis['Ef (eV/atom)'],
                        colorscale='Viridis',
                        size=5,
                        symbol='circle',
                        colorbar=dict(title='E<sub>f</sub> (eV/atom)'),
                    ),
                )
            )
            # Add labels for the five elements
            fig = set_labels_plotly(fig, labels)
            fig.update_layout(
                title='UMAP Projection of Predicted Formation Energy:',
                **kwargs_fig
            )
            st.plotly_chart(fig, use_container_width=False)

        # Lattice Distortion
        with col2:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(**kwargs_data_all))
            fig.add_trace(
                go.Scatter(
                    **kwargs_data_dis,
                    marker=dict(
                        color=data_dis['D_lattice (cm^2/s)'],
                        colorscale='Viridis',
                        size=5,
                        symbol='circle',
                        colorbar=dict(title=r'Δ<sub>lattice</sub> (cm²/s)'),
                    ),
                )
            )
            # Add labels for the five elements
            fig = set_labels_plotly(fig, labels)
            fig.update_layout(
                title='UMAP Projection of Predicted Lattice Distortion:',
                **kwargs_fig
            )
            st.plotly_chart(fig, use_container_width=False)

        # Atomic Distortion
        with col3:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(**kwargs_data_all))
            fig.add_trace(
                go.Scatter(
                    **kwargs_data_dis,
                    marker=dict(
                        color=data_dis['D_atomic (cm^2/s)'],
                        colorscale='Viridis',
                        size=5,
                        symbol='circle',
                        colorbar=dict(title='Δ<sub>atomic</sub> (cm²/s)'),
                    ),
                )
            )
            # Add labels for the five elements
            fig = set_labels_plotly(fig, labels)
            fig.update_layout(
                title='UMAP Projection of Predicted Atomic Distortion:',
                **kwargs_fig
            )
            st.plotly_chart(fig, use_container_width=False)

        # Diffusivity
        with col4:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(**kwargs_data_all))
            fig.add_trace(
                go.Scatter(
                    **kwargs_data_dis,
                    marker=dict(
                        color=data_dis['D (cm^2/s)'],
                        colorscale='Viridis',
                        size=5,
                        symbol='circle',
                        colorbar=dict(title='D (cm²/s)'),
                    ),
                )
            )
            # Add labels for the five elements
            fig = set_labels_plotly(fig, labels)
            fig.update_layout(
                title='UMAP Projection of Predicted Diffusion Coefficient:',
                **kwargs_fig
            )
            st.plotly_chart(fig, use_container_width=False)
    


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