'''
 # @ Create Time: 2023-10-23 15:17:02.807444
'''

import numpy as np
import pandas as pd
import plotly.express as px
import dash
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Initialize the app
app = dash.Dash('HH_interactive')

# Declare server for Heroku deployment. Needed for Procfile.
server = app.server

#import data
HH = pd.read_csv('HH_Beta_Tetrahedral_temp.csv')
sub_columns = HH[['Compound','kappa_L','beta_n','DOSmass_CB','bmass_CB','mobility_n','Nb_CB','beta_p','DOSmass_VB','bmass_VB','mobility_p','Nb_VB']]
sub_columns_n = HH[['Compound','beta_n','DOSmass_CB','bmass_CB','mobility_n','kappa_L','Nb_CB']]
sub_columns_p = HH[['Compound','beta_p','DOSmass_VB','bmass_VB','mobility_p','kappa_L','Nb_VB']]
sub_columns.dropna()
sub_columns_n.dropna()
sub_columns_p.dropna()

#prep data to numeric#ensure n type data is numeric
sub_columns_n['beta_n'] = pd.to_numeric(sub_columns_n['beta_n'])
sub_columns_n['mobility_n'] = pd.to_numeric(sub_columns_n['mobility_n'])
sub_columns_n['bmass_CB'] = pd.to_numeric(sub_columns_n['bmass_CB'])
sub_columns_n['kappa_L'] = pd.to_numeric(sub_columns_n['kappa_L'])
sub_columns_n['DOSmass_CB'] = pd.to_numeric(sub_columns_n['DOSmass_CB'])
sub_columns_n['Nb_CB'] = pd.to_numeric(sub_columns_n['Nb_CB'])

#ensure p type data is numeric
sub_columns_p['beta_p'] = pd.to_numeric(sub_columns_p['beta_p'])
sub_columns_p['mobility_p'] = pd.to_numeric(sub_columns_p['mobility_p'])
sub_columns_p['bmass_VB'] = pd.to_numeric(sub_columns_p['bmass_VB'])
sub_columns_p['DOSmass_VB'] = pd.to_numeric(sub_columns_p['DOSmass_VB'])
sub_columns_p['Nb_VB'] = pd.to_numeric(sub_columns_p['Nb_VB'])
sub_columns_p['kappa_L'] = pd.to_numeric(sub_columns_p['kappa_L'])

#ensure common data is numeric
sub_columns['beta_p'] = pd.to_numeric(sub_columns['beta_p'])
sub_columns['mobility_p'] = pd.to_numeric(sub_columns['mobility_p'])
sub_columns['bmass_VB'] = pd.to_numeric(sub_columns['bmass_VB'])
sub_columns['DOSmass_VB'] = pd.to_numeric(sub_columns['DOSmass_VB'])
sub_columns['Nb_VB'] = pd.to_numeric(sub_columns['Nb_VB'])
sub_columns['kappa_L'] = pd.to_numeric(sub_columns['kappa_L'])
sub_columns['beta_n'] = pd.to_numeric(sub_columns['beta_n'])
sub_columns['mobility_n'] = pd.to_numeric(sub_columns['mobility_n'])
sub_columns['bmass_CB'] = pd.to_numeric(sub_columns['bmass_CB'])
sub_columns['DOSmass_CB'] = pd.to_numeric(sub_columns['DOSmass_CB'])
sub_columns['Nb_CB'] = pd.to_numeric(sub_columns['Nb_CB'])

#invert band masses and kappas
bmass_n = np.array(sub_columns_n['bmass_CB'])
bmass_p = np.array(sub_columns_p['bmass_VB'])

#common
bmass_n_common = np.array(sub_columns['bmass_CB'])
bmass_p_common = np.array(sub_columns['bmass_VB'])

kappa_L = np.array(sub_columns_n['kappa_L'])
kappa_common = np.array(sub_columns['kappa_L'])

sub_columns_n['1/bmass_CB'] = np.reciprocal(bmass_n)
sub_columns_n['1/kappa_L'] = np.reciprocal(kappa_L)
sub_columns_p['1/bmass_VB'] = np.reciprocal(bmass_p)
sub_columns_p['1/kappa_L'] = np.reciprocal(kappa_L)

sub_columns['1/bmass_CB'] = np.reciprocal(bmass_n_common)
sub_columns['1/kappa_L'] = np.reciprocal(kappa_common)
sub_columns['1/bmass_VB'] = np.reciprocal(bmass_p_common)

n_type_data =  sub_columns_n.drop(['bmass_CB','kappa_L'],axis=1)
p_type_data = sub_columns_p.drop(['bmass_VB','kappa_L'],axis=1)
common_data = sub_columns.drop(['bmass_CB','bmass_VB','kappa_L'],axis=1)

properties = ['beta','mobility','band_mass','thermal_conductivity','DOS_mass','band_degeneracy']

#corresponding column names
col_names_n = {'beta':'beta_n','mobility':'mobility_n','band_mass':'1/bmass_CB','thermal_conductivity':'1/kappa_L','DOS_mass':'DOSmass_CB','band_degeneracy':'Nb_CB'}
col_names_p = {'beta':'beta_p','mobility':'mobility_p','band_mass':'1/bmass_VB','thermal_conductivity':'1/kappa_L','DOS_mass':'DOSmass_VB','band_degeneracy':'Nb_VB'}

#reverse names for plotting
#chart_names_n = {'beta_n':'beta','mobility_n':'mobility','1/bmass_CB':'band mass','kappa_L':}
#chart_names_p = {}

# Create a dropdown list of column names (excluding 'Compound')
dropdown_options = [{'label': prop, 'value': prop} for prop in properties]


fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'bar'}]])


# Set up the app layout
app.layout = html.Div([
    html.H1("Half-Heusler Semiconductors Interactive"),
    dcc.Dropdown(
        id='column-dropdown',
        options=dropdown_options,
        value=dropdown_options[0]['value'],
        style={'width': '50%'}
    ),
    dcc.Input(
        id='num-entries-input',
        type='number',
        value=5,
        placeholder='Enter number of entries',
        style={'width': '50%'}
    ),
    dcc.Graph(
        id='average-graph1',
        figure=go.Figure()  # Initial empty figure for the first chart
    ),
    dcc.Graph(
        id='average-graph2',
        figure=go.Figure()  # Initial empty figure for the second chart
    ),
    dcc.Graph(
    id='average-graph3',
    figure=go.Figure()  # Initial empty figure for the third chart
    ),
    html.Div([
        html.H3('n-type Compounds'),
        html.Ol(id='list-output1')  # New Div for the list output
    ]),
    html.Div([
        html.H3('p-type Compounds'),
        html.Ol(id='list-output2')  # New Div for the list output
    ]),
    html.Div([
    html.H3('Common n and p-type Compounds'),
    html.Ol(id='list-output3')  # New Div for the list output
    ])
])
                  
# Define the callback to update the graphs
@app.callback(
    [Output('average-graph1', 'figure'),
     Output('average-graph2', 'figure'),
     Output('average-graph3', 'figure'),
     Output('list-output1', 'children'),
     Output('list-output2', 'children'),
     Output('list-output3', 'children')],
    [Input('column-dropdown', 'value'),
     Input('num-entries-input', 'value')]
)

# Define the callback function
def update_graph(selected_column, num_entries):
    
    #make sub_df of n best numbers
    cur_best_n = n_type_data.nlargest(num_entries,col_names_n[selected_column])
    cur_best_p = p_type_data.nlargest(num_entries,col_names_p[selected_column])
    
    #common best
    common_compounds = list(set(list(cur_best_n.Compound)).intersection(cur_best_p.Compound))
    common_df = common_data[common_data['Compound'].isin(common_compounds)]
    
    # Calculate the average of all columns for the selected column
    
    n_means = cur_best_n.describe().loc['mean']
    n_maxes = cur_best_n.describe().loc['max']
    n_perc_maxed = (n_means/n_maxes)*100
    
    p_means = cur_best_p.describe().loc['mean']
    p_maxes = cur_best_p.describe().loc['max']
    p_perc_maxed = (p_means/p_maxes)*100
    
    common_means = common_df.describe().loc['mean']
    common_maxes = common_df.describe().loc['max']
    common_perc_maxed = (common_means/common_maxes)*100

    #format means for hover labels
    n_hover = n_means
    p_hover = p_means
    common_hover = common_means

    n_hover['1/bmass_CB'] = np.reciprocal(float(n_hover['1/bmass_CB']))
    n_hover['1/kappa_L'] = np.reciprocal(float(n_hover['1/kappa_L']))
    n_hover = np.around(np.array(n_hover.values),3)
    p_hover['1/bmass_VB'] = np.reciprocal(float(p_hover['1/bmass_VB']))
    p_hover['1/kappa_L'] = np.reciprocal(float(p_hover['1/kappa_L']))
    p_hover = np.around(np.array(p_hover.values),3)
    common_hover['1/bmass_VB'] = np.reciprocal(float(common_hover['1/bmass_VB']))
    common_hover['1/bmass_CB'] = np.reciprocal(float(common_hover['1/bmass_CB']))
    common_hover['1/kappa_L'] = np.reciprocal(float(common_hover['1/kappa_L']))
    common_hover = np.around(np.array(common_hover.values),3)

    # Create first bar chart
    bar_chart1 = go.Figure(go.Bar(x=list(n_perc_maxed.index), y=list(n_perc_maxed.values),hovertext= n_hover))
    bar_chart1.update_layout(title=f'n-type Properties With Best {num_entries} {selected_column} chosen ', barmode='group',yaxis = dict(title = 'Percent of Maximum Value Reached'))
    
    # Create second bar chart (you can customize this as needed)
    bar_chart2 = go.Figure(go.Bar(x=list(p_perc_maxed.index), y=list(p_perc_maxed.values), marker_color='orange',hovertext = p_hover))
    bar_chart2.update_layout(title=f'p-type Properties With Best {num_entries} {selected_column} chosen ', barmode='group',yaxis = dict(title = 'Percent of Maximum Value Reached'))

    #create third common bar chart
    bar_chart3 = go.Figure(go.Bar(x=list(common_perc_maxed.index), y=list(common_perc_maxed.values), marker_color='lightgreen',hovertext = common_hover))
    bar_chart3.update_layout(title=f'Overall Properties for Common n and p-type Compounds Given Best {num_entries} {selected_column} chosen ', barmode='group',yaxis = dict(title = 'Percent of Maximum Value Reached'))

    #output compounds formatted for HTML
    text_n = list(cur_best_n.Compound)
    list_n = html.Ol([html.Li(item) for item in text_n])
    text_p = list(cur_best_p.Compound)
    list_p = html.Ol([html.Li(item) for item in text_p])
    text_common = list(common_df.Compound)
    list_common = html.Ol([html.Li(item) for item in text_common])
    
    return bar_chart1, bar_chart2,bar_chart3,list_n, list_p,list_common

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

