import dash
import dash_core_components as dcc
from dash import html
from dash import dcc
import numpy as np
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output, State
import io
import base64
import dash_bootstrap_components as dbc
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV


from sklearn.model_selection import train_test_split
from scipy.stats import linregress
from sklearn.metrics import mean_absolute_error
import plotly.graph_objs as go
import pmdarima as pm
from pmdarima.arima import auto_arima


#int(df.index[-1])

#Quellen: Hirschle S. 52 zb
#Quelle auto arima Train test split 118. hirschle

#App erstellen, wobei ich hier auf das Design 'dbc.themes.COSMO' zurückgreife
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.COSMO], suppress_callback_exceptions=True)
server = app.server

#layout ist ein HTML-Element, das den graphischen Aufbau der Seite festlegt.
layout = html.Div([
    html.H1(children='Forecasting im Controlling', 
            style={
                'textAlign': 'center',
                'width': '100%',
                'height': '100px',
                'backgroundColor': '#0B2896',
                'lineHeight': '90px',
                'color': '#ffffff'
                }),
    html.Div(style={'height': '50px'}),
    html.Div([
        html.Div(
            #an dieser Stelle wird der Button 'CSV-Datei hochladen' spezifiziert
            dcc.Upload(
                id='upload-data',
                children=html.Div([html.A('CSV-Datei hochladen')]),
                style={
                    'width': '20%',
                    'height': '80px',
                    'lineHeight': '70px',
                    'backgroundColor': '#0B2896',
                    'borderWidth': '5px',
                    'borderStyle': 'solid' ,
                    'borderColor': 'grey',
                    'borderRadius': '30px',
                    'textAlign': 'center',
                    'margin': '0 auto',
                    'color': '#ffffff',
                    'font-size': '30px'
                },
                multiple=False
            ),
            style={ 'textAlign': 'center', 'width': '100%'}
        ),
        html.Div(html.P(children=['Mit einem Klick auf den Button, können Sie eine CSV-Datei auswählen. Diese sollte aus zwei Spalten bestehen. Die erste Spalte sollte aus Datumsangaben (chronologisch sortiert) beinhalten.',html.Br(),'Die zweite Spalte sollte aus den zugehörigen, numerischen Werten bestehen. Sie können die hochgeladene Datei jederzeit über den Button ändern oder Ihre Eingabe rückgängig machen, indem Sie die Seite neu laden.'], style={'font-size': '17px', 'textAlign': 'center', 'margin': '20px' }
    )),
        html.Div(html.P(children=['Anzahl der Perioden, die prognostiziert werden sollen:',html.Br()], style={'font-size': '20px', 'textAlign': 'center', 'margin': '20px', 'color': '#0B2896'}
    ))
        ]),
            html.Div(
        style={
            'display': 'flex',
            'justify-content': 'center',
            'align-items': 'center'
        },
        children=[        
            dcc.Dropdown(
                #Dropdown-Menü, das festlegt, wie viele zusätzliche Perioden des Datensatzes vorhergesagt werden sollen. Per Default ist eine Periode eingestellt.
                id='periods-dropdown',            
                options=[{'label': '1', 'value': '1'},
                        {'label': '2', 'value': '2'}],
                    value='1',
                    style={
                        'width': '40%',
                        'height': '50px',
                        'font-size': '20px',
                        'borderWidth': '3px',
            
                        'textAlign': 'center',
                        'margin': '0 auto'
                    }
            )
        ]
    ),
    html.Div(style={'height': '50px'}),
    html.Div([
        html.Div(style={'width': '5%', 'height': '900px'}),
        html.Div([
            dcc.Dropdown(
                #Dropdown-Menü mit herkömmlichen Prognosemethoden. Per Default ist die lineare Regression eingestellt.
                id='dropdown1',
                options=[
                    {'label': 'lineare Regression', 'value': 'linReg'},
                    {'label': 'Moving Average', 'value': 'movingAverage'},
                    {'label': 'Exponential Smoothing', 'value': 'exponentialSmoothing'}
           
                ],
                #später wieder hinzufügen
                #value='linReg',
                value = 'linReg',
                style={'font-size': '20px', 'borderWidth': '3px', 'margin': '10px'}
            ),
           
            html.Div([
                #Unterhalb des Dropdowns soll das Liniendiagramm angezeigt werden.
                html.Div(id='output-data-upload-links', style={'width': '100%', 'display': 'inline-block'})
            ]),
            html.Div(id='textContainerLinks', style={'font-size': '20px'})
            #an dieser Stelle soll der Text unterhalb des Diagramms angegeben werden
        ], className='col', style={'float': 'left', 'width': '42.5%', 'backgroundColor': '#ffffff', 'borderStyle': 'solid' ,
                    'borderColor': '#ffffff',
                    'borderRadius': '15px','borderWidth': '5px', 'height': '1500px'}),
        html.Div(style={'width': '5%', 'height': '900px'}),
        html.Div([
            dcc.Dropdown(
                #Dropdown-Menü mit den Prognosemethoden, die auf Machine Learning Basieren. Per Default ist die lineare Regression eingestellt.
                id='dropdown2',
                options=[
                    {'label': 'lineare Regression mit Gewichtungsfunktion', 'value': 'linearRegression'},
                    {'label': 'Auto-ARIMA', 'value': 'autoArima'},
                    {'label': 'Ridge Cross Validation', 'value': 'ridgeCV'}
                ],
                #value='linearRegression',
                value = 'linearRegression',
                style={'font-size': '20px', 'borderWidth': '3px', 'margin': '10px'}
                
            ),
            html.Div([
                #Unterhalb des Dropdown soll das Liniendiagramm erscheinen
                html.Div(id='output-data-upload', style={'width': '100%', 'display': 'inline-block'})
            ]),
            html.Div(id='textContainerRechts', style={'font-size': '20px'})
            #an dieser Stelle soll der Text unterhalb des Diagramms angegeben werden
        ], className='col', style={'float': 'right', 'width': '42.5%', 'backgroundColor': '#ffffff', 'borderStyle': 'solid' ,
                    'borderColor': '#ffffff',
                    'borderRadius': '15px','borderWidth': '5px', 'height': '1500px'}),
        html.Div(style={'width': '5%', 'height': '900px'})
    ], className='row'), 
    
], style={'backgroundColor': '#F0F5F7', 'height': '3000px'})

#lesen der CSV-Datei und Weiterverarbeitung als Pandas-Dataframe (tabellarisch)
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    if 'csv' in filename:
        df = pd.read_csv(
            io.StringIO(decoded.decode('utf-8')))
        return df
    else:
        return None

######## Liste mit den Mean Absolute Errors oder Abweichungsarrays ######

mae_links = [0, 0, 0]
mae_rechts = [0, 0, 0]

mae_links2 = [0, 0, 0]
mae_rechts2 = [0, 0, 0]

abweichung_links = [0,0,0]
abweichung_rechts = [0,0,0]

proz_abweichung_links = [0,0,0]

    
###### MA 3 #######################################

def moving_average(data, ordnung):
    #Berechnet den gleitenden Durchschnitt für die gegebene Datenreihe mit der gegebenen Ordnung (hier k=3).
    window = np.ones(int(ordnung)) / float(ordnung)
    return np.convolve(data, window, 'same')

############linke Seite#############################  

@app.callback(
    #die Callback-Funktion spezifiziert notwendige Inputs, die die untenstehende Funktion update_output2 aktivieren, sowie was als Output ausgegeben wird. Darüberhinaus spzifiziert State den Zustand. Ändert sich dieser, wird die Funktion erneut ausgeführt.
    Output('output-data-upload-links', 'children'),
    Input('dropdown1', 'value'),
    Input('periods-dropdown', 'value'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output2(selected_graph, periods, list_of_contents, list_of_names):
    if selected_graph == 'linReg'  and periods == '1':
        if list_of_contents is not None:
            contents = list_of_contents
            filename = list_of_names
            df = parse_contents(contents, filename)
            if df is None:
                return html.Div(['Die Datei scheint leer oder keine gültige CSV-Datei zu sein.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px'
                                
                                })
            df = df.dropna()
            last_index = int(df.index[-1])
            print('Datentyp: ',type(last_index))
            df = df.append({df.columns[0]: (last_index + 1)}, ignore_index=True)
            
            #der Datensatz darf nicht leer sein
            if df is not None:
                #leere Zeilen des Datensatzes löschen, damit die Regression durchgeführt werden kann
                df2 = df.dropna()
                #X sei die Spalte mit den Datumsangaben und Y Spalte mit den zugehörigen Werten
                #x = df2.iloc[:, [0]].values.flatten()
                x= df2.index.to_frame().values.flatten()
                y = df2.iloc[:, [1]].values.flatten()
                
                #Durchführung der linearen Regression
                slope, intercept, r_value, p_value, std_err = linregress(x, y)
                nextPrediction = slope * (last_index + 1) + intercept

                
                if nextPrediction < 0:
                        df.iloc[:, 1][df.index[-1]] = 0.0
                else:  
                    df.iloc[:, 1][df.index[-1]] = nextPrediction
            
               
                df3 = df.iloc[:-1]
                df3 = df3.dropna()
                #x2 = df3.iloc[:, [0]].values.flatten()
                x2 = df3.index[:-1].to_frame().values.flatten()
                #y2 = df3.iloc[:, [1]].values.flatten()
                y2 = df3.iloc[:, [1]].values.flatten()[:-1]
                
                #da hier eine Berechnung des MAE nicht möglich ist, nehme ich einen Datenpunkt weg, sage ihn vorher und berechne auf Grundlage darauf den MAE
                slope2, intercept2, r_value2, p_value2, std_err2 = linregress(x2, y2)
                actual_value = y[-1]
                print('LEAS FRAGE, actual_value:', actual_value)
                print('last-index:', last_index)
                next_prediction2 = slope2 * (last_index) + intercept2
                print('LEAS FRAGE, next_prediction2:', next_prediction2)
                
                mae = mean_absolute_error([actual_value], [next_prediction2])
                mae = mae.round(2)
                mae_links[0] = mae
                

                print('LEA:', [next_prediction2])
                print('LEA:', [next_prediction2][0])
         
                
                proz_abweichung = (abs([next_prediction2][0] - [actual_value][0]) / [actual_value][0]) * 100
                proz_abweichung =  proz_abweichung.round(2)
                proz_abweichung_links[0] = proz_abweichung
                
                df = df.round(2)
                
                #Nun soll der Graph definiert werden :)
                x_values = [str(i) for i in df.index]
                fig = px.line(title='klassische lineare Regression')
                
                fig.add_trace(go.Scatter(
                    #x=df.iloc[:-1, 0],
                    x=x_values[:-1],
                    y=df.iloc[:-1, 1],
                    mode="lines",
                    line=dict(color= '#4A4AE8'),
                    name="Daten"
                ))

                fig.add_trace(go.Scatter(
                    x=[x_values[-2] , x_values[-1]],
                    y=[df.iloc[-2, 1], df.iloc[-1, 1]],                 
                    mode="lines",
                    line=dict(dash="dash", color="red"),
                    name="Vorhersage"
                ))


                fig.update_layout(

                    yaxis=dict(
                        title="Wert"
                    ),
                    xaxis=dict(
                        title="Zeit"
                    )
) 


                return dcc.Graph(
                    id='linregression-plot',
                    figure=fig
                )
  
        else:
            return html.Div(['Bitte gültige CSV-Datei hochladen.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px',
                                'margin': '40px'
                                
                                })
    
    elif selected_graph == 'linReg' and periods == '2':
        if list_of_contents is not None:
            contents = list_of_contents
            filename = list_of_names
            df = parse_contents(contents, filename)
            
            if df is None:
                return html.Div(['Die Datei scheint leer oder keine gültige CSV-Datei zu sein.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px'
                                
                                })            
            
            df = df.dropna()
            last_index = int(df.index[-1])
            df = df.append({df.columns[0]: (last_index + 1)}, ignore_index=True)
            df = df.append({df.columns[0]: (last_index + 2)}, ignore_index=True)

            if df is not None:
                df2 = df.dropna()
                #X sei die Spalte mit den Datumsangaben und Y die zugehörigen WErte
                #x = df2.iloc[:, [0]].values.flatten()
                x= df2.index.to_frame().values.flatten()
                y = df2.iloc[:, [1]].values.flatten()
        
                slope, intercept, r_value, p_value, std_err = linregress(x, y)
                nextPrediction = slope * (last_index + 1) + intercept
                nextPrediction2 = slope * (last_index + 2) + intercept

                
                if nextPrediction < 0:
                        df.iloc[:, 1][df.index[-2]] = 0.0
                else:  
                    df.iloc[:, 1][df.index[-2]] = nextPrediction

                if nextPrediction2 < 0:
                        df.iloc[:, 1][df.index[-1]] = 0.0
                else:  
                    df.iloc[:, 1][df.index[-1]] = nextPrediction2          
               
                df3 = df.iloc[:-2]
                df3 = df3.dropna()
                #x2 = df3.iloc[:, [0]].values.flatten()
                #y2 = df3.iloc[:, [1]].values.flatten()
                x2 = df3.index[:-1].to_frame().values.flatten()
                #y2 = df3.iloc[:, [1]].values.flatten()
                y2 = df3.iloc[:, [1]].values.flatten()[:-1]
                
                #da hier eine Berechnung des MAE nicht möglich ist, nehme ich einen Datenpunkt weg, sage ihn vorher und berechne auf Grundlage darauf den MAE
                slope2, intercept2, r_value2, p_value2, std_err2 = linregress(x2, y2)
                actual_value = y[-1]
                next_prediction2 = slope2 * (last_index) + intercept2
                
                mae = mean_absolute_error([actual_value], [next_prediction2])
                mae = mae.round(2)
                mae_links2[0] = mae
                
                df = df.round(2)
                
                #Nun soll der Graph definiert werden :)
                x_values = [str(i) for i in df.index]
                fig = px.line(title='klassische lineare Regression')
                
                fig.add_trace(go.Scatter(
                    #x=df.iloc[:-1, 0],
                    x=x_values[:-1],
                    y=df.iloc[:-1, 1],
                    mode="lines",
                    line=dict(color= '#4A4AE8'),
                    name="Daten"
                ))

                fig.add_trace(go.Scatter(
                    #probiere wegzumachen
                    #x=[df.iloc[-2, 0], df.iloc[-1, 0]],
                    x=[x_values[-2], x_values[-1]],
                    y=[df.iloc[-2, 1], df.iloc[-1, 1]],
                    mode="lines",
                    line=dict(dash="dash", color="red"),
                    name="Vorhersage"
                ))

                fig.update_layout(

                    yaxis=dict(
                        title="Wert"
                    ),
                    xaxis=dict(
                        title="Zeit"
                    )
) 
                return dcc.Graph(
                    id='linregression-plot1',
                    figure=fig
                )
    

        else:
            return html.Div(['Bitte gültige CSV-Datei hochladen.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px',
                                'margin': '40px'
                                
                                })
    
    elif selected_graph == 'exponentialSmoothing' and periods == '1':
            
        if list_of_contents is not None:
            contents = list_of_contents
            filename = list_of_names
            df = parse_contents(contents, filename)
            
            if df is None:
                return html.Div(['Die Datei scheint leer oder keine gültige CSV-Datei zu sein.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px'
                                
                                })            
            
            df = df.dropna()
            last_index = int(df.index[-1])
            df = df.append({df.columns[0]: (last_index + 1)}, ignore_index=True)

            if df is not None:
                df2 = df.dropna()

                x = df2.iloc[:, [0]].values.flatten()
                y = df2.iloc[:, [1]].values.flatten()
                
                #Exponential Smoothing berechnen
                alpha = 0.2
                smoothed_values = [y[0]]
                for i in range(1, len(y)):
                    smoothed_value = alpha * y[i] + (1 - alpha) * smoothed_values[i-1]
                    smoothed_values.append(smoothed_value)
                
                nextPrediction = smoothed_values[-1]
                print('smoothed_value', smoothed_values)
                print('nextpred:', nextPrediction)
         
                
                mae = mean_absolute_error(y[-1:], smoothed_values[-2:-1])
                mae = mae.round(2)
                mae_links[1] = mae
                
                if nextPrediction < 0:
                    df.iloc[:, 1][df.index[-1]] = 0.0
                else:  
                    df.iloc[:, 1][df.index[-1]] = nextPrediction  
                    
                df = df.round(2) 
                    
                #Nun soll der Graph definiert werden :)
                x_values = [str(i) for i in df.index]
                fig = px.line(title='Exponential Smoothing')
                
                fig.add_trace(go.Scatter(
                    #x=df.iloc[:-1, 0],
                    x=x_values[:-1],
                    y=df.iloc[:-1, 1],
                    mode="lines",
                    line=dict(color= '#4A4AE8'),
                    name="Daten"
                ))

                fig.add_trace(go.Scatter(
                    #probiere wegzumachen
                    #x=[df.iloc[-2, 0], df.iloc[-1, 0]],
                    x=[x_values[-2], x_values[-1]],
                    y=[df.iloc[-2, 1], df.iloc[-1, 1]],
                    mode="lines",
                    line=dict(dash="dash", color="red"),
                    name="Vorhersage"
                ))

                fig.update_layout(

                    yaxis=dict(
                        title="Wert"
                    ),
                    xaxis=dict(
                        title="Zeit"
                    )
) 


                return dcc.Graph(
                    id='exponentialsmoothing-plot',
                    figure=fig
                )

        else:
            return html.Div(['Bitte gültige CSV-Datei hochladen.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px',
                                'margin': '40px'
                                
                                })

    elif selected_graph == 'exponentialSmoothing' and periods == '2':
            
        if list_of_contents is not None:
            contents = list_of_contents
            filename = list_of_names
            df = parse_contents(contents, filename)
            
            if df is None:
                return html.Div(['Die Datei scheint leer oder keine gültige CSV-Datei zu sein.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px'
                                
                                })            
            
            df = df.dropna()
            print(df.iloc[:, 0])
            #Index auf 0 setzen
            last_index = int(df.index[-1])
            print('LastIndex')
            print(last_index)
            df = df.append({df.columns[0]: (last_index + 1)}, ignore_index=True)
            df = df.append({df.columns[0]: (last_index + 2)}, ignore_index=True)
            print('????????????')
            print(df)
            if df is not None:
                df2 = df.dropna()

               

                return ''


        else:
            return html.Div(['Bitte gültige CSV-Datei hochladen.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px',
                                'margin': '40px'
                                
                                })
            
           
    elif selected_graph == 'movingAverage' and periods == '1':
            
        if list_of_contents is not None:
            contents = list_of_contents
            filename = list_of_names
            df = parse_contents(contents, filename)
            
            if df is None:
                return html.Div(['Die Datei scheint leer oder keine gültige CSV-Datei zu sein.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px'
                                
                                })            
            
            df = df.dropna()
            last_index = int(df.index[-1])
            df = df.append({df.columns[0]: (last_index + 1)}, ignore_index=True)

            if df is not None:
                df2 = df.dropna()

                x = df2.iloc[:, [0]].values.flatten()
                y = df2.iloc[:, [1]].values.flatten()
                
                #Moving Average der Ordnung drei berechnen
                moving_avg = moving_average(y, 3)
                print('MOVING AVERAGE: ', moving_avg)
                nextPrediction = moving_avg[-2] 
                
                print('nextPrediction: ', nextPrediction)
                print('LEA tatsächlicher wert ma:', moving_avg[-3:-2])
                
                #mae = mean_absolute_error(y[-1:], moving_avg[-3:])
                mae = mean_absolute_error(y[-1:], moving_avg[-3:-2])
                mae = mae.round(2)
                mae_links[2] = mae
                
                if nextPrediction < 0:
                    df.iloc[:, 1][df.index[-1]] = 0.0
                else:  
                    df.iloc[:, 1][df.index[-1]] = nextPrediction 
                
                print(df)  
                
                df = df.round(2)
                    
                #Nun soll der Graph definiert werden :)
                x_values = [str(i) for i in df.index]
                fig = px.line(title='Moving Average (MA3)')
                
                fig.add_trace(go.Scatter(
                    #x=df.iloc[:-1, 0],
                    x=x_values[:-1],
                    y=df.iloc[:-1, 1],
                    mode="lines",
                    line=dict(color= '#4A4AE8'),
                    name="Daten"
                ))

                fig.add_trace(go.Scatter(
                    #probiere wegzumachen
                    #x=[df.iloc[-2, 0], df.iloc[-1, 0]],
                    x=[x_values[-2], x_values[-1]],
                    y=[df.iloc[-2, 1], df.iloc[-1, 1]],
                    mode="lines",
                    line=dict(dash="dash", color="red"),
                    name="Vorhersage"
                ))

                fig.update_layout(

                    yaxis=dict(
                        title="Wert"
                    ),
                    xaxis=dict(
                        title="Zeit"
                    )
) 


                return dcc.Graph(
                    id='movingaverage-plot',
                    figure=fig
                )
    

        else:
            return html.Div(['Bitte gültige CSV-Datei hochladen.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px',
                                'margin': '40px'
                                
                                })  
    elif selected_graph == 'movingAverage' and periods == '2':
            
        if list_of_contents is not None:
            contents = list_of_contents
            filename = list_of_names
            df = parse_contents(contents, filename)
            
            if df is None:
                return html.Div(['Die Datei scheint leer oder keine gültige CSV-Datei zu sein.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px'
                                
                                })

            df = df.dropna()
            print(df.iloc[:, 0])
            #Index auf 0 setzen
            last_index = int(df.index[-1])
            print('LastIndex')
            print(last_index)
            df = df.append({df.columns[0]: (last_index + 1)}, ignore_index=True)
            df = df.append({df.columns[0]: (last_index + 2)}, ignore_index=True)
            print('????????????')
            print(df)
            if df is not None:
                df2 = df.dropna()

                x = df2.iloc[:, [0]].values.flatten()
                y = df2.iloc[:, [1]].values.flatten()
                
                #Moving Average der Ordnung 3 berechnen
                moving_avg = moving_average(y, 3)
                nextPrediction = moving_avg[-1] 
                y = np.append(y, nextPrediction)
                moving_avg2 = moving_average(y, 3)
                nextPrediction2 = moving_avg2[-1]                

                
                mae = mean_absolute_error(y[-3:], moving_avg[-3:])
                mae = mae.round(2)
                mae_links2[2] = mae
                
                if nextPrediction < 0:
                    df.iloc[:, 1][df.index[-2]] = 0.0
                else:  
                    df.iloc[:, 1][df.index[-2]] = nextPrediction  

                if nextPrediction2 < 0:
                    df.iloc[:, 1][df.index[-1]] = 0.0
                else:  
                    df.iloc[:, 1][df.index[-1]] = nextPrediction2    
                    
                #Nun soll der Graph definiert werden :)
                fig = px.line(title='Moving Average (MA3)')
                
                fig.add_trace(go.Scatter(
                    x=df.iloc[:-2, 0],
                    y=df.iloc[:-2, 1],
                    mode="lines",
                    line=dict(color= '#4A4AE8'),
                    name="Daten"
                ))

                fig.add_trace(go.Scatter(
                    x=[df.iloc[-3, 0],df.iloc[-2, 0], df.iloc[-1, 0]],
                    y=[df.iloc[-3, 1],df.iloc[-2, 1], df.iloc[-1, 1]],
                    mode="lines",
                    line=dict(dash="dash", color="red"),
                    name="Vorhersage"
                ))

                fig.update_layout(
                    xaxis=dict(
                        title="Zeit",
                        tickmode="linear",
                        tick0=df.iloc[0, 0],
                        dtick=2,
                        range=[df.iloc[0, 0], df.iloc[-1, 0]],
                    ),
                    yaxis=dict(
                        title="Wert"
                    )
                )


                return ''
    

        else:
            return html.Div(['Bitte gültige CSV-Datei hochladen.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px',
                                'margin': '40px'
                                
                                })  

        



        

############rechte Seite###########################       
@app.callback(
    Output('output-data-upload', 'children'),
    Input('dropdown2', 'value'),
    Input('periods-dropdown', 'value'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(selected_graph, periods, list_of_contents, list_of_names):
    if selected_graph == 'linearRegression' and periods =='1':

        if list_of_contents is not None:
            contents = list_of_contents
            filename = list_of_names
            df = parse_contents(contents, filename)

            if df is None:
                return html.Div(['Die Datei scheint leer oder keine gültige CSV-Datei zu sein.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px'
                                
                                })

            df = df.dropna()
            last_index = int(df.index[-1])
            df = df.append({df.columns[0]: (last_index + 1)}, ignore_index=True)

            if df is not None:
                df2 = df.dropna()
                #X sei die Spalte mit den Datumsangaben und Y die zugehörigen WErte
                x = df2.index.to_frame()
                y = df2.iloc[:, [1]]

        
                x_train, x_test, y_train, y_test = train_test_split(x,y, random_state= 0, test_size=0.2)
                x_train.sort_index(inplace=True)
                y_train.sort_index(inplace=True)
            
                indizes = x_train.index.tolist()

                gewichte = []
                anzahlDatenpunkte = len(df2) 

                for x in indizes:
                    #TriCube Gewichtungsfunktion
                    d = (abs(x_train.loc[x] - x_train.iloc[-1, 0])) / anzahlDatenpunkte
                    weight = pow((1- pow(d,3)),3)
                    weight=weight.round(2)
                    weight = weight[0]
                    
                    gewichte.append(weight)
        
                model= LinearRegression()
                model.fit(x_train, y_train, gewichte)
                y_predict = model.predict(x_test)
                
                mae = mean_absolute_error(y_test, y_predict)
                mae = mae.round(2)
                mae_rechts[0] = mae
                
                nextPrediction = model.predict([[(last_index + 1)]]).round(2)
                print('Prediction')
                print(nextPrediction)
                

            
                #um die Vergleichbarkeit mit den herkömmlichen Forecasting-Methoden (ohne Machine Learning)
                #sicherzustellen, wird der letze Datenpunkt der Zeitreihe extrahiert, eine lineare Regr. berechnet
                #und anschließend, die Höhe des prognostizierten Werts mit dem extrahierten Wert verglichen
                
                #start
                #x2 = df2.iloc[:-1, [0]]
                x2 = df2.index[:-1].to_frame()
                y2 = df2.iloc[:-1, [1]]

                x_train, x_test, y_train, y_test = train_test_split(x2, y2, random_state=0, test_size=0.2)
                x_train.sort_index(inplace=True)
                y_train.sort_index(inplace=True)

                indizes = x_train.index.tolist()

                gewichte = []
                anzahlDatenpunkte = len(df2) - 1
                
                for x in indizes:
                    #TriCube Gewichtungsfunktion
                    d = (abs(x_train.loc[x] - x_train.iloc[-1, 0])) / anzahlDatenpunkte
                    weight = pow((1- pow(d,3)),3)
                    weight=weight.round(2)
                    weight = weight[0]
                    
                    gewichte.append(weight)

                model = LinearRegression()
                model.fit(x_train, y_train, gewichte)
                last_value = df2.index[-1]
                verkürztePrognose = model.predict([[last_value]])
                #verkürztePrognose = model.predict(df2.iloc[[-1], [0]])
                print('verkürztePrognose: ', verkürztePrognose)
            
                
                abweichung = mean_absolute_error(y[-1:], verkürztePrognose)
                abweichung = abweichung.round(2)
                abweichung_rechts[0] = abweichung
      
                
                ######
                
                if nextPrediction < 0:
                        df.iloc[:, 1][df.index[-1]] = 0.0
                else:  
                    df.iloc[:, 1][df.index[-1]] = nextPrediction
                
                df = df.round(2)
                #Nun soll der Graph definiert werden :)
                x_values = [str(i) for i in df.index]
                fig = px.line(title='Lineare Regression (mit Machine Learning)')
                
                fig.add_trace(go.Scatter(
                    #x=df.iloc[:-1, 0],
                    x=x_values[:-1],
                    y=df.iloc[:-1, 1],
                    mode="lines",
                    line=dict(color= '#4A4AE8'),
                    name="Daten"
                ))

                fig.add_trace(go.Scatter(
                    #probiere wegzumachen
                    #x=[df.iloc[-2, 0], df.iloc[-1, 0]],
                    x=[x_values[-2], x_values[-1]],
                    y=[df.iloc[-2, 1], df.iloc[-1, 1]],
                    mode="lines",
                    line=dict(dash="dash", color="red"),
                    name="Vorhersage"
                ))

                fig.update_layout(

                    yaxis=dict(
                        title="Wert"
                    ),
                    xaxis=dict(
                        title="Zeit"
                    )
)

                return dcc.Graph(
                    id='regression-plot',
                    figure=fig
                )
    

        else:
            return html.Div(['Bitte gültige CSV-Datei hochladen.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px',
                                'margin': '40px'
                                
                                })
         
    elif selected_graph == 'linearRegression' and periods =='2':

        if list_of_contents is not None:
            contents = list_of_contents
            filename = list_of_names
            df = parse_contents(contents, filename)

            if df is None:
                return html.Div(['Die Datei scheint leer oder keine gültige CSV-Datei zu sein.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px'
                                
                                })
            
            df = df.dropna()
            
            last_index = int(df.index[-1])
            df = df.append({df.columns[0]: (last_index + 1)}, ignore_index=True)
            df = df.append({df.columns[0]: (last_index + 2)}, ignore_index=True)

            if df is not None:
                df2 = df.dropna()
                x = df2.index.to_frame()
                y = df2.iloc[:, [1]]
                
                x_train, x_test, y_train, y_test = train_test_split(x,y, random_state= 0, test_size=0.2)
                x_train.sort_index(inplace=True)
                y_train.sort_index(inplace=True)

                #model= LinearRegression()
                #model.fit(x_train, y_train)
                #y_predict = model.predict(x_test)

                indizes = x_train.index.tolist()

                gewichte = []
                anzahlDatenpunkte = len(df2) 
                for x in indizes:
                    #TriCube Gewichtungsfunktion
                    d = (abs(x_train.loc[x] - x_train.iloc[-1, 0])) / anzahlDatenpunkte
                    weight = pow((1- pow(d,3)),3)
                    weight=weight.round(2)
                    weight = weight[0]
                    
                    gewichte.append(weight)
                
                model= LinearRegression()
                model.fit(x_train, y_train, gewichte)
                y_predict = model.predict(x_test)

                #das nächste Jahr wird vorhergesagt
                prediction1 = model.predict([[(last_index + 1)]]).round(2)
                prediction2 = model.predict([[(last_index + 2)]]).round(2)


                if prediction1 < 0:
                    df.iloc[:, 1][df.index[-2]] = 0.0
                else:
                    df.iloc[:, 1][df.index[-2]] = prediction1 

                if prediction2 < 0:
                    df.iloc[:, 1][df.index[-1]] = 0.0
                else:
                    df.iloc[:, 1][df.index[-1]] = prediction2 
                
                
                mae = mean_absolute_error(y_test, y_predict)
                mae = mae.round(2)
                mae_rechts[0] = mae
                
                #
                df = df.round(2)
                #Nun soll der Graph definiert werden :)
                x_values = [str(i) for i in df.index]
                fig = px.line(title='Lineare Regression (mit Machine Learning)')
                
                fig.add_trace(go.Scatter(
                    #x=df.iloc[:-1, 0],
                    x=x_values[:-1],
                    y=df.iloc[:-1, 1],
                    mode="lines",
                    line=dict(color= '#4A4AE8'),
                    name="Daten"
                ))

                fig.add_trace(go.Scatter(
                    #probiere wegzumachen
                    #x=[df.iloc[-2, 0], df.iloc[-1, 0]],
                    x=[x_values[-2], x_values[-1]],
                    y=[df.iloc[-2, 1], df.iloc[-1, 1]],
                    mode="lines",
                    line=dict(dash="dash", color="red"),
                    name="Vorhersage"
                ))

                fig.update_layout(

                    yaxis=dict(
                        title="Wert"
                    ),
                    xaxis=dict(
                        title="Zeit"
                    )
)                 

                return dcc.Graph(
                    id='regression-plot',
                    figure=fig
                )
    

        else:
            return html.Div(['Bitte gültige CSV-Datei hochladen.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px',
                                'margin': '40px'
                                
                                })
            
    elif selected_graph == 'autoArima' and periods == '1':
        if list_of_contents is not None:
            contents = list_of_contents
            filename = list_of_names
            df = parse_contents(contents, filename)
            
            if df is None:
                return html.Div(['Die Datei scheint leer oder keine gültige CSV-Datei zu sein.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px'
                                
                                })
                        
            df = df.dropna()
            print(df.iloc[:, 0])
            #Index auf 0 setzen
            last_index = int(df.index[-1])
            print('LastIndex')
            print(last_index)
            df = df.append({df.columns[0]: (last_index + 1)}, ignore_index=True)

            if df is not None:
  
                df2 = df.dropna()
                df2= df2.drop(df2.columns[2:], axis=1)
                print('df2:', df2)

                
                column_names = df2.columns.tolist()
                column_index = 0
                column_name = column_names[column_index]


                df2.iloc[:, [column_index]] = pd.to_datetime(df2.iloc[:, [column_index]].astype(str).agg('-'.join, axis=1), format='%d.%m.%Y')
                df2 = df2.set_index(column_name)
                
                grenze = 0.8 * round(len(df2))
                grenze = int(grenze)
                train = df2[:grenze]
                test = df2[grenze:]
                print('test', test)
                print('train', train)


                #Auto-Arima trainieren
                arima_model = auto_arima(train, start_p=0, d=1, start_q=0, test='adf', max_p=5, max_d=5, max_q=5, start_P=0, D=1, start_Q=0,
                                    max_P=5, max_D=2, max_Q=5, m = 4, seasonal=True, stepwise=True, random_state=20, n_fits=50, 
                                    suppress_warnings=True, trace=True)
       
                #Vorhersage der Testreihen: 2023-01-01   972.222780
                #es wird immer ab test predicted
                length_prediction = int(len(test))
                prediction = pd.DataFrame(arima_model.predict(length_prediction))
                print('FUCKING prediction:', prediction)

                mae = mean_absolute_error(test, prediction)
                mae = mae.round(2)
                mae_rechts[1] = mae

                nextPrediction = arima_model.predict(length_prediction + 1)
                nextPrediction = nextPrediction[-1]
                nextPrediction = round(nextPrediction, 2)     
     
                
                ##########
                #um die Vergleichbarkeit mit den herkömmlichen Forecasting-Methoden (ohne Machine Learning)
                #sicherzustellen, wird der letze Datenpunkt der Zeitreihe extrahiert, eine lineare Regr. berechnet
                #und anschließend, die Höhe des prognostizierten Werts mit dem extrahierten Wert verglichen
                
                #start
                df3 = df2.iloc[:-1]
               
                grenze = 0.8 * round(len(df3))
                grenze = int(grenze)
                train2 = df3[:grenze]
                test2 = df3[grenze:]
                print('test2: ', test2)
                zahl = int(len(test2))

                #Auto-Arima trainieren
                arima_model = auto_arima(train2, start_p=0, d=1, start_q=0, test='adf', max_p=5, max_d=5, max_q=5, start_P=0, D=1, start_Q=0,
                                    max_P=5, max_D=2, max_Q=5, m =4, seasonal=True, stepwise=True, random_state=20, n_fits=50, 
                                    suppress_warnings=True, trace=True, n_periods=zahl)
                

                length_prediction = int(len(test2))
                nextPrediction2 = pd.DataFrame(arima_model.predict(length_prediction + 1))
                print('LEA PREDICTION2', nextPrediction2)
                #2023-01-01  1040.882669
                nextPrediction2 = nextPrediction2.iloc[-1, 0]
                nextPrediction2 = nextPrediction2.round(2)

                
                abweichung = mean_absolute_error(test[-1:], [nextPrediction2])
                abweichung = abweichung.round(2)
                abweichung_rechts[1] = abweichung

                #########

                if nextPrediction < 0:
                        df.iloc[:, 1][df.index[-1]] = 0.0
                else:  
                    df.iloc[:, 1][df.index[-1]] = nextPrediction
            
                print('PredictionARIMA', nextPrediction)
                
                
                
                #Nun soll der Graph definiert werden :)
                df = df.round(2)
                x_values = [str(i) for i in df.index]
                fig = px.line(title='Auto-ARIMA')
                
                fig.add_trace(go.Scatter(
                    #x=df.iloc[:-1, 0],
                    x=x_values[:-1],
                    y=df.iloc[:-1, 1],
                    mode="lines",
                    line=dict(color= '#4A4AE8'),
                    name="Daten"
                ))

                fig.add_trace(go.Scatter(
                    #probiere wegzumachen
                    #x=[df.iloc[-2, 0], df.iloc[-1, 0]],
                    x=[x_values[-2], x_values[-1]],
                    y=[df.iloc[-2, 1], df.iloc[-1, 1]],
                    mode="lines",
                    line=dict(dash="dash", color="red"),
                    name="Vorhersage"
                ))

                fig.update_layout(

                    yaxis=dict(
                        title="Wert"
                    ),
                    xaxis=dict(
                        title="Zeit"
                    )
)    

                return dcc.Graph(
                    id='arima-plot',
                    figure=fig
                )
    

         
        else:
            return html.Div(['Bitte gültige CSV-Datei hochladen.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px',
                                'margin': '40px'
                                
                                })     

    elif selected_graph == 'autoArima' and periods == '2':
        if list_of_contents is not None:
            contents = list_of_contents
            filename = list_of_names
            df = parse_contents(contents, filename)
            if df is None:
                return html.Div(['Die Datei scheint leer oder keine gültige CSV-Datei zu sein.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px'
                                
                                })
            df = df.dropna()

            #Index auf 0 setzen
            last_index = int(df.index[-1])

            df = df.append({df.columns[0]: (last_index + 1)}, ignore_index=True)
            df = df.append({df.columns[0]: (last_index + 2)}, ignore_index=True)
 
            if df is not None:
  
                df2 = df.dropna()
                df2= df2.drop(df2.columns[2:], axis=1)
                print('df2:', df2)
                #X sei die Spalte mit den Datumsangaben und Y die zugehörigen WErte
                #x = df2.iloc[:, [0]]
                #y = df2.iloc[:, [1]].values
                
                column_names = df2.columns.tolist()
                column_index = 0
                column_name = column_names[column_index]


                df2.iloc[:, [column_index]] = pd.to_datetime(df2.iloc[:, [column_index]].astype(str).agg('-'.join, axis=1), format='%d.%m.%Y')
                
                
                df2 = df2.set_index(column_name)
                
                grenze = 0.8 * round(len(df2))
                grenze = int(grenze)
                train = df2[:grenze]
                test = df2[grenze:]
                print('test', test)
                print('train', train)
                


                #Auto-Arima trainieren
                arima_model = auto_arima(train, start_p=0, d=1, start_q=0, test='adf', max_p=5, max_d=5, max_q=5, start_P=0, D=1, start_Q=0,
                                    max_P=5, max_D=2, max_Q=5, m =1, seasonal=False, stepwise=True, random_state=20, n_fits=50, 
                                    suppress_warnings=True, trace=True)
#
                length_prediction = int(len(test))
                prediction = pd.DataFrame(arima_model.predict(length_prediction))
                print('FUCKING prediction:', prediction)

                nextPrediction = arima_model.predict(length_prediction + 2)
                nextPrediction2 = nextPrediction[-1]
                nextPrediction2 = round(nextPrediction2, 2) 

                nextPrediction1 = nextPrediction[-2]
                nextPrediction1 = round(nextPrediction1, 2) 
                print('Vorletzte:', nextPrediction1)
                print('letzte:', nextPrediction2)
#
                


                
                #2023-04-01    1036.412851
                if nextPrediction1 < 0:
                        df.iloc[:, 1][df.index[-2]] = 0.0
                else:  
                    df.iloc[:, 1][df.index[-2]] = nextPrediction1

                
                if nextPrediction2 < 0:
                        df.iloc[:, 1][df.index[-1]] = 0.0
                else:  
                    df.iloc[:, 1][df.index[-1]] = nextPrediction2
                
 
                #Nun soll der Graph definiert werden :)
                df = df.round(2)
                x_values = [str(i) for i in df.index]
                fig = px.line(title='Auto-ARIMA')
                
                fig.add_trace(go.Scatter(
                    #x=df.iloc[:-1, 0],
                    x=x_values[:-1],
                    y=df.iloc[:-1, 1],
                    mode="lines",
                    line=dict(color= '#4A4AE8'),
                    name="Daten"
                ))

                fig.add_trace(go.Scatter(
                    #probiere wegzumachen
                    #x=[df.iloc[-2, 0], df.iloc[-1, 0]],
                    x=[x_values[-2], x_values[-1]],
                    y=[df.iloc[-2, 1], df.iloc[-1, 1]],
                    mode="lines",
                    line=dict(dash="dash", color="red"),
                    name="Vorhersage"
                ))

                fig.update_layout(

                    yaxis=dict(
                        title="Wert"
                    ),
                    xaxis=dict(
                        title="Zeit"
                    )
)    



                return dcc.Graph(
                    id='arima-plot',
                    figure=fig
                )
    


        else:
            return html.Div(['Bitte gültige CSV-Datei hochladen.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px',
                                'margin': '40px'
                                
                                })     
        
        
    elif selected_graph == 'ridgeCV' and periods == '1':
        if list_of_contents is not None:
            contents = list_of_contents
            filename = list_of_names
            df = parse_contents(contents, filename)
            
            if df is None:
                return html.Div(['Die Datei scheint leer oder keine gültige CSV-Datei zu sein.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px'
                                
                                })
            
            df = df.dropna()
            # Index auf 0 setzen
            last_index = int(df.index[-1])
            df = df.append({df.columns[0]: (last_index + 1)}, ignore_index=True)

            if df is not None:
                df2 = df.dropna()
                x = df2.index.to_frame()
                print('**********+', x)
                y = df2.iloc[:, [1]]
                
                myalpha = np.linspace(start = 0.1,stop = 5,num = 50)
                x_train, x_test, y_train, y_test = train_test_split(x,y, random_state= 0, test_size=0.2)
                
                ridge =RidgeCV(alphas = myalpha)
                ridge.fit(x_train, np.array(y_train).ravel())
                y_predict = ridge.predict(x_test)

            
                mae = mean_absolute_error(y_test, y_predict)
                mae = mae.round(2)
                mae_rechts[2] = mae
                
                print('ridgeMAE:', mae)
                
                nextPrediction = ridge.predict([[(last_index + 1)]]).round(2)
                print('Prediction')
                print(nextPrediction)

            
                #um die Vergleichbarkeit mit den herkömmlichen Forecasting-Methoden (ohne Machine Learning)
                #sicherzustellen, wird der letze Datenpunkt der Zeitreihe extrahiert, eine lineare Regr. berechnet
                #und anschließend, die Höhe des prognostizierten Werts mit dem extrahierten Wert verglichen
                
                #start
                x2 = df2.index[:-1].to_frame()
                y2 = df2.iloc[:-1, [1]]

                x_train, x_test, y_train, y_test = train_test_split(x2, y2, random_state=0, test_size=0.2)
 
                ridge =RidgeCV(alphas = myalpha)
                ridge.fit(x_train, np.array(y_train).ravel())
                #verkürztePrognose = ridge.predict(df2.iloc[[-1], [0]])
                last_value = df2.index[-1]
                verkürztePrognose = ridge.predict([[last_value]])
                print('verkürztePrognose', verkürztePrognose)
    

                abweichung = mean_absolute_error(y[-1:], verkürztePrognose)
                abweichung = abweichung.round(2)
                abweichung_rechts[2] = abweichung
      
                
                ######
                
                if nextPrediction < 0:
                        df.iloc[:, 1][df.index[-1]] = 0.0
                else:  
                    df.iloc[:, 1][df.index[-1]] = nextPrediction

                #Nun soll der Graph definiert werden :)
                df = df.round(2)
                x_values = [str(i) for i in df.index]
                fig = px.line(title='Ridge Cross Validation')
                
                fig.add_trace(go.Scatter(
                    #x=df.iloc[:-1, 0],
                    x=x_values[:-1],
                    y=df.iloc[:-1, 1],
                    mode="lines",
                    line=dict(color= '#4A4AE8'),
                    name="Daten"
                ))

                fig.add_trace(go.Scatter(
                    #probiere wegzumachen
                    #x=[df.iloc[-2, 0], df.iloc[-1, 0]],
                    x=[x_values[-2], x_values[-1]],
                    y=[df.iloc[-2, 1], df.iloc[-1, 1]],
                    mode="lines",
                    line=dict(dash="dash", color="red"),
                    name="Vorhersage"
                ))

                fig.update_layout(

                    yaxis=dict(
                        title="Wert"
                    ),
                    xaxis=dict(
                        title="Zeit"
                    )
)    



                return dcc.Graph(
                    id='ridgeCV-plot',
                    figure=fig
                )
    

        else:
            return html.Div(['Bitte gültige CSV-Datei hochladen.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px',
                                'margin': '40px'
                                
                                })

    elif selected_graph == 'ridgeCV' and periods == '2':
        if list_of_contents is not None:
            contents = list_of_contents
            filename = list_of_names
            df = parse_contents(contents, filename)
            
            if df is None:
                return html.Div(['Die Datei scheint leer oder keine gültige CSV-Datei zu sein.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px'
                                
                                })
            
            df = df.dropna()
            # Index auf 0 setzen
            last_index = int(df.index[-1])
            df = df.append({df.columns[0]: (last_index + 1)}, ignore_index=True)
            df = df.append({df.columns[0]: (last_index + 2)}, ignore_index=True)

            if df is not None:
                df2 = df.dropna()
                x = df2.index.to_frame()
                y = df2.iloc[:, [1]]
                
                myalpha = np.linspace(start = 0.1,stop = 5,num = 50)

                x_train, x_test, y_train, y_test = train_test_split(x,y, random_state= 0, test_size=0.2)
                
                ridge =RidgeCV(alphas = myalpha)
                ridge.fit(x_train, np.array(y_train).ravel())
                y_predict = ridge.predict(x_test)
            
                mae = mean_absolute_error(y_test, y_predict)
                mae = mae.round(2)
                mae_rechts[2] = mae
                
                print('ridgeMAE:', mae)
                


                #Nun soll der Graph definiert werden :)
                df = df.round(2)
                x_values = [str(i) for i in df.index]
                fig = px.line(title='Ridge Cross Validation')
                
                fig.add_trace(go.Scatter(
                    #x=df.iloc[:-1, 0],
                    x=x_values[:-1],
                    y=df.iloc[:-1, 1],
                    mode="lines",
                    line=dict(color= '#4A4AE8'),
                    name="Daten"
                ))

                fig.add_trace(go.Scatter(
                    #probiere wegzumachen
                    #x=[df.iloc[-2, 0], df.iloc[-1, 0]],
                    x=[x_values[-2], x_values[-1]],
                    y=[df.iloc[-2, 1], df.iloc[-1, 1]],
                    mode="lines",
                    line=dict(dash="dash", color="red"),
                    name="Vorhersage"
                ))

                fig.update_layout(

                    yaxis=dict(
                        title="Wert"
                    ),
                    xaxis=dict(
                        title="Zeit"
                    )
) 


                return dcc.Graph(
                    id='ridgeCV-plot2',
                    figure=fig
                )
    


        else:
            return html.Div(['Bitte gültige CSV-Datei hochladen.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px',
                                'margin': '40px'
                                
                                })

############# Text Container Rechts ########################
@app.callback(
    Output('textContainerRechts', 'children'),
    Input('dropdown2', 'value'),
    Input('periods-dropdown', 'value'),
    #Input('output-data-upload-links', 'children'),
    Input('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
    
)

def update_dropdown2_text(selected, periods, graph, x, y):
    if selected == 'linearRegression' and periods == '1':

        if x is not None:
            contents = x
            filename = y
            df = parse_contents(contents, filename)

            if df is None:
                return ''
       
            else: 
                return  html.P(children=[
                    'Die ',
                    html.Strong('lineare Regression'),
                    ' der Python-Bibliothek ', html.Em('Scikit-Learn'), 'teilt den Datensatz mit deren Spalten (bestehend aus 2 Spalten: eine für die Datums/Jahreszahlangabe und eine weitere für die zugehörigen Werte (z.B. Umsatz oder ähnliche Kennzahlen)) in Trainings-',
                    'und Testdaten auf. Etwa 20 Prozent des Datensatzes werden als Testdaten zurückbehalten. Anschließend wird die lineare Regression basierend auf den Trainingsdaten trainiert und ist nun in der Lage, die nächsten Datenpunkte für die folgenden Perioden vorherzusagen. ',
                    html.Br(), html.Br(), 'In diesem Fall wird ein Mean-Absolute-Error in Höhe von ', html.Strong(str(mae_rechts[0])), ' erzielt',
                    html.Br(), 'Dieser Mean-Absolute Error (kurz: MAE) gibt an, inwieweit die vorhergesagten Datenpunkte von den tatsächlichen Werten des Testdatensatzes abweicht. Je näher der MAE an der 0 liegt, desto besser schneidet das Machine-Learning-Modell (hier: die lineare Regression) bei der Vorhersage ab.',
                    html.Br(), html.Br(), 'Zur besseren Vergleichbarkeit mit den traditionellen Prognosemodellen, wird die letzte Datenreihe im weiteren Durchgang extrahiert, noch einmal eine lineare Regression durchgeführt und der extrahierte Wert prognostiziert. Somit lässt sich erkennen, wie nah der prognostizierte Wert am tatsächlichen Wert liegt. Die Abweichung lautet für diesen Datensatz: ', html.Strong(str(abweichung_rechts[0])), '.'
                ])
    if selected == 'autoArima' and periods == '1':
        if x is not None:
            contents = x
            filename = y
            df = parse_contents(contents, filename)

            if df is None:
                return ''
            else: return html.P(children=[
            'Das ',
            html.Strong('Auto-ARIMA'),
            ' Modell der Python-Bibliothek ', html.Em('pmdarima'), ' teilt den Datensatz mit deren Spalten (bestehend aus 2 Spalten: eine für die Datums/Jahreszahlangabe und eine weitere für die zugehörigen Werte (z.B. Umsatz oder ähnliche Kennzahlen)) in Trainings-',
            'und Testdaten auf. Etwa 20 Prozent des Datensatzes werden als Testdaten zurückbehalten. Anschließend wird automatisiert ein ARIMA Zeitreihenmodell aufgestellt und verschiedene Durchgänge mit diversen Parametern trainiert. Das Modell mit den besten Werten gewinnt und ist nun in der Lage, die nächsten Datenpunkte für die folgenden Perioden vorherzusagen. ',
            html.Br(), html.Br(), 'In diesem Fall wird ein Mean-Absolute-Error in Höhe von ', html.Strong(str(mae_rechts[1])), ' erzielt',
            html.Br(), 'Dieser Mean-Absolute Error (kurz: MAE) gibt an, inwieweit die vorhergesagten Datenpunkte von den tatsächlichen Werten des Testdatensatzes abweicht.',
            html.Br(), html.Br(), 'Zur besseren Vergleichbarkeit mit den traditionellen Prognosemodellen, wird die letzte Datenreihe im weiteren Durchgang extrahiert, noch einmal eine Auto-ARIMA-Prognose durchgeführt und der extrahierte Wert prognostiziert. Somit lässt sich erkennen, wie nah der prognostizierte Wert am tatsächlichen Wert liegt. Die Abweichung lautet für diesen Datensatz: ', html.Strong(str(abweichung_rechts[1])), '.'
        ])
    if selected == 'ridgeCV' and periods == '1':
        if x is not None:
            contents = x
            filename = y
            df = parse_contents(contents, filename)

            if df is None:
                return ''
            else:
                return html.P(children=[
                    'Das ',
                    html.Strong('Ridge Regression mit Cross Validation'),
                    ' der Python-Bibliothek ', html.Em('Scikit-Learn'), 'teilt den Datensatz mit deren Spalten (bestehend aus 2 Spalten: eine für die Datums/Jahreszahlangabe und eine weitere für die zugehörigen Werte (z.B. Umsatz oder ähnliche Kennzahlen)) in Trainings-',
                    'und Testdaten auf. Etwa 20 Prozent des Datensatzes werden als Testdaten zurückbehalten. Anschließend werden die besten Parameter automatisiert mithilfe der Cross Validation für die Ridge Regression ermittelt. Das Modell mit den besten Werten gewinnt und ist nun in der Lage, die nächsten Datenpunkte für die folgenden Perioden vorherzusagen. ',
                    html.Br(), html.Br(), 'In diesem Fall wird ein Mean-Absolute-Error in Höhe von ', html.Strong(str(mae_rechts[2])), ' erzielt',
                    html.Br(), 'Dieser Mean-Absolute Error (kurz: MAE) gibt an, inwieweit die vorhergesagten Datenpunkte von den tatsächlichen Werten des Testdatensatzes abweicht. Der MAE kann Werte zwischen 0 und 1 annehmen. Je näher er an der 0 liegt, desto besser schneidet das Machine-Learning-Modell (hier: die lineare Regression) bei der Vorhersage ab. Je näher der MAE an der 1 liegt, desto schlechter die Performance des Modells',
                    html.Br(), html.Br(), 'Zur besseren Vergleichbarkeit mit den traditionellen Prognosemodellen, wird die letzte Datenreihe im weiteren Durchgang extrahiert, noch einmal eine Ridge Regression mit Cross Validation durchgeführt und der extrahierte Wert prognostiziert. Somit lässt sich erkennen, wie nah der prognostizierte Wert am tatsächlichen Wert liegt. Die Abweichung lautet für diesen Datensatz: ', html.Strong(str(abweichung_rechts[2])), '.'
                ])

    if selected == 'linearRegression' and periods == '2':
        if x is not None:
            contents = x
            filename = y
            df = parse_contents(contents, filename)

            if df is None:
                return ''
            else:
                return  html.P(children=[
                    'Die ',
                    html.Strong('lineare Regression'),
                    ' der Python-Bibliothek ', html.Em('Scikit-Learn'), 'teilt den Datensatz mit deren Spalten (bestehend aus 2 Spalten: eine für die Datums/Jahreszahlangabe und eine weitere für die zugehörigen Werte (z.B. Umsatz oder ähnliche Kennzahlen)) in Trainings-',
                    'und Testdaten auf. Etwa 20 Prozent des Datensatzes werden als Testdaten zurückbehalten. Anschließend wird die lineare Regression basierend auf den Trainingsdaten trainiert und ist nun in der Lage, die nächsten Datenpunkte für die folgenden Perioden vorherzusagen. ',
                    html.Br(), html.Br(), 'In diesem Fall wird ein Mean-Absolute-Error in Höhe von ', html.Strong(str(mae_rechts[0])), ' erzielt',
                    html.Br(), 'Dieser Mean-Absolute Error (kurz: MAE) gibt an, inwieweit die vorhergesagten Datenpunkte von den tatsächlichen Werten des Testdatensatzes abweicht. Der MAE kann Werte zwischen 0 und 1 annehmen. Je näher er an der 0 liegt, desto besser schneidet das Machine-Learning-Modell (hier: die lineare Regression) bei der Vorhersage ab. Je näher der MAE an der 1 liegt, desto schlechter die Performance des Modells',
                    html.Br(), html.Br(), 'Zur besseren Vergleichbarkeit mit den traditionellen Prognosemodellen, wird die letzte Datenreihe im weiteren Durchgang extrahiert, noch einmal eine lineare Regression durchgeführt und der extrahierte Wert prognostiziert. Somit lässt sich erkennen, wie nah der prognostizierte Wert am tatsächlichen Wert liegt. Die Abweichung lautet für diesen Datensatz: ', html.Strong(str(abweichung_rechts[0])), '.'
                ])
    if selected == 'autoArima' and periods == '2':
        if x is not None:
            contents = x
            filename = y
            df = parse_contents(contents, filename)

            if df is None:
                return ''
            else:
                return html.P(children=[
                    'Das ',
                    html.Strong('Auto-ARIMA'),
                    ' Modell der Python-Bibliothek ', html.Em('pmdarima'), ' teilt den Datensatz mit deren Spalten (bestehend aus 2 Spalten: eine für die Datums/Jahreszahlangabe und eine weitere für die zugehörigen Werte (z.B. Umsatz oder ähnliche Kennzahlen)) in Trainings-',
                    'und Testdaten auf. Etwa 20 Prozent des Datensatzes werden als Testdaten zurückbehalten. Anschließend wird automatisiert ein ARIMA Zeitreihenmodell aufgestellt und verschiedene Durchgänge mit diversen Parametern trainiert. Das Modell mit den besten Werten gewinnt und ist nun in der Lage, die nächsten Datenpunkte für die folgenden Perioden vorherzusagen. ',
                    html.Br(), html.Br(), 'In diesem Fall wird ein Mean-Absolute-Error in Höhe von ', html.Strong(str(mae_rechts[1])), ' erzielt',
                    html.Br(), 'Dieser Mean-Absolute Error (kurz: MAE) gibt an, inwieweit die vorhergesagten Datenpunkte von den tatsächlichen Werten des Testdatensatzes abweicht. Der MAE kann Werte zwischen 0 und 1 annehmen. Je näher er an der 0 liegt, desto besser schneidet das Machine-Learning-Modell (hier: die lineare Regression) bei der Vorhersage ab. Je näher der MAE an der 1 liegt, desto schlechter die Performance des Modells',
                    html.Br(), html.Br(), 'Zur besseren Vergleichbarkeit mit den traditionellen Prognosemodellen, wird die letzte Datenreihe im weiteren Durchgang extrahiert, noch einmal eine Auto-ARIMA-Vorhersage durchgeführt und der extrahierte Wert prognostiziert. Somit lässt sich erkennen, wie nah der prognostizierte Wert am tatsächlichen Wert liegt. Die Abweichung lautet für diesen Datensatz: ', html.Strong(str(abweichung_rechts[1])), '.'
                ])
    if selected == 'ridgeCV' and periods == '2':
        if x is not None:
            contents = x
            filename = y
            df = parse_contents(contents, filename)

            if df is None:
                return ''
            else:
                return html.P(children=[
                    'Das ',
                    html.Strong('Ridge Regression mit Cross Validation'),
                    ' der Python-Bibliothek ', html.Em('Scikit-Learn'), 'teilt den Datensatz mit deren Spalten (bestehend aus 2 Spalten: eine für die Datums/Jahreszahlangabe und eine weitere für die zugehörigen Werte (z.B. Umsatz oder ähnliche Kennzahlen)) in Trainings-',
                    'und Testdaten auf. Etwa 20 Prozent des Datensatzes werden als Testdaten zurückbehalten. Anschließend werden die besten Parameter automatisiert mithilfe der Cross Validation für die Ridge Regression ermittelt. Das Modell mit den besten Werten gewinnt und ist nun in der Lage, die nächsten Datenpunkte für die folgenden Perioden vorherzusagen. ',
                    html.Br(), html.Br(), 'In diesem Fall wird ein Mean-Absolute-Error', html.Sup('1'), 'in Höhe von ', html.Strong(str(mae_rechts[2])), ' erzielt',
                    html.Br(), 'Dieser Mean-Absolute Error (kurz: MAE) gibt an, inwieweit die vorhergesagten Datenpunkte von den tatsächlichen Werten des Testdatensatzes abweicht. Der MAE kann Werte zwischen 0 und 1 annehmen. Je näher er an der 0 liegt, desto besser schneidet das Machine-Learning-Modell (hier: die lineare Regression) bei der Vorhersage ab. Um Vergleichbarkeit XXX ', html.Strong(str(abweichung_rechts[2])),
                    html.Br(), html.Br(), 'Zur besseren Vergleichbarkeit mit den traditionellen Prognosemodellen, wird die letzte Datenreihe im weiteren Durchgang extrahiert, noch einmal eine Ridge Regression mit Cross Validation durchgeführt und der extrahierte Wert prognostiziert. Somit lässt sich erkennen, wie nah der prognostizierte Wert am tatsächlichen Wert liegt. Die Abweichung lautet für diesen Datensatz: ', html.Strong(str(abweichung_rechts[2])), '.',
                    html.Br(),
                    html.Br(),
                        html.Div(children=[
                            html.Sup('1'),
                            ' Vgl. XXX (XXX), S. XXX.'
                        ])
                ])

    else:
        return ''
############# Text Container Links ########################

@app.callback(
    Output('textContainerLinks', 'children'),
    Input('dropdown1', 'value'),
    Input('periods-dropdown', 'value'),
    Input('output-data-upload-links', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)

def update_dropdown2_text(selected, periods, graph, x, y):
    if selected == 'linReg' and periods =='1':
        if x is not None:
            contents = x
            filename = y
            df = parse_contents(contents, filename)

            if df is None:
                return ''
            else:
       
                return html.P(children=[
                    'Dieser Graph zeigt die Ergebnisse der ',
                    html.Strong('klassischen linearen Regression'),
                    ' ,welche mithilfe der Python-Bibliothek Scipy durchgeführt wird. ',html.Br(), html.Br(), ' Die blaue Linie kennzeichnet die tatsächlichen Werte der eingelesenen Zeitreihe, wohingegen die rote Linie die Vorhersage zeigt. Um die Güte dieses Modells vergleichen zu können, wird abermals eine lineare Regression ohne die letzte Datenreihe durchgeführt und der extrahierte Wert prognostiziert. Anschließend kann mit der Differenz des tatsächlichen Werts zum prognostizierten Wert beurteilt werden, wie gut das Modell bei der Vorhersage abschneidet.',html.Br(), 'In diesem Fall ergibt sich eine Abweichung in Höhe von: ', html.Strong(str(mae_links[0])), '. Je höher diese Abweichung, desto schlechter ist das Modell in der Lage auf Basis des eingelesenen Datensatzes eine Vorhersage zu treffen.',
                    html.Br(), 'Des Weiteren ergibt sich eine prozentuale Abweichung des prognostizierten Werts zum realen Wert in Höhe von: ', html.Strong(str(proz_abweichung_links[0]))
                    
                ])
        
    if selected == 'exponentialSmoothing' and periods =='1':
        if x is not None:
            contents = x
            filename = y
            df = parse_contents(contents, filename)

            if df is None:
                return ''
            else:
       
                return html.P(children=[
                    'Dieser Graph zeigt die Ergebnisse des ',
                    html.Strong('Expontential Smoothing'),
                    ' (Exponentielle Glättung). ',html.Br(), html.Br(), ' Die blaue Linie kennzeichnet die tatsächlichen Werte der eingelesenen Zeitreihe, wohingegen die rote Linie die Vorhersage zeigt. Um die Güte dieses Modells vergleichen zu können, wird abermals eine lineare Regression ohne die letzte Datenreihe durchgeführt und der extrahierte Wert prognostiziert. Anschließend kann mit der Differenz des tatsächlichen Werts zum prognostizierten Wert beurteilt werden, wie gut das Modell bei der Vorhersage abschneidet.',html.Br(), 'In diesem Fall ergibt sich eine Abweichung in Höhe von: ', html.Strong(str(mae_links[1])), '. Je höher diese Abweichung, desto schlechter ist das Modell in der Lage auf Basis des eingelesenen Datensatzes eine Vorhersage zu treffen.',
                    html.Br(), html.Br(), 'Es ist zu beachten, dass das Exponential Smoothing vergangenen Entwicklungen hinterherhinkt und somit in manchen Fällen bei der Abweichungsanalyse vermeintlich besser abschneidet als andere Modelle.'
                ])
    if selected == 'movingAverage'and periods =='1':
        if x is not None:
            contents = x
            filename = y
            df = parse_contents(contents, filename)

            if df is None:
                return ''
            else:
                return html.P(children=[
                    'Dieser Graph zeigt die Ergebnisse des ',
                    html.Strong('Moving Average Modell der dritten Ordnung'),
                    ' . Das heißt basierend auf den jeweils drei vorherigen Datenpunkten wird der nächste Vorhergesagt.',
                    html.Br(), html.Br(), ' Die blaue Linie kennzeichnet die tatsächlichen Werte der eingelesenen Zeitreihe, wohingegen die rote Linie die Vorhersage zeigt. Um die Güte dieses Modells vergleichen zu können, wird abermals eine lineare Regression ohne die letzte Datenreihe durchgeführt und der extrahierte Wert prognostiziert. Anschließend kann mit der Differenz des tatsächlichen Werts zum prognostizierten Wert beurteilt werden, wie gut das Modell bei der Vorhersage abschneidet.',html.Br(), 'In diesem Fall ergibt sich eine Abweichung in Höhe von: ', html.Strong(str(mae_links[2])), '. Je höher diese Abweichung, desto schlechter ist das Modell in der Lage auf Basis des eingelesenen Datensatzes eine Vorhersage zu treffen.',
                    html.Br(), html.Br(), 'Es ist zu beachten, dass das Moving Average Verfahren der dritten Ordnung vergangenen Entwicklungen hinterherhinkt und somit in manchen Fällen bei der Abweichungsanalyse vermeintlich besser abschneidet als andere Modelle.'
                ])

    if selected == 'linReg' and periods =='2':
        if x is not None:
            contents = x
            filename = y
            df = parse_contents(contents, filename)

            if df is None:
                return ''
            else:
       
                return html.P(children=[
                    'Dieser Graph zeigt die Ergebnisse der ',
                    html.Strong('klassischen linearen Regression'),
                    ' ,welche mithilfe der Python-Bibliothek Scipy durchgeführt wird. ',html.Br(), html.Br(), 'Um die Güte dieses Modells vergleichen zu können, wird abermals eine lineare Regression ohne die letzte Datenreihe durchgeführt und der extrahierte Wert prognostiziert. Anschließend kann mit der Differenz des tatsächlichen Werts zum prognostizierten Wert beurteilt werden, wie gut das Modell bei der Vorhersage abschneidet. In diesem Fall ergibt sich eine Abweichung in Höhe von: ', html.Strong(str(mae_links[0])), '.',
                    
                ])
        
    if selected == 'exponentialSmoothing' and periods =='2':
        if x is not None:
            contents = x
            filename = y
            df = parse_contents(contents, filename)

            if df is None:
                return ''
            else:
       
                return html.P(children=[
                    'Dieser Graph zeigt die Ergebnisse des ',
                    html.Strong('Expontential Smoothing'),
                    ' (Exponentielle Glättung),welches mithilfe der Python-Bibliothek Scipy durchgeführt wird. ',html.Br(), html.Br(), 'Um die Güte dieses Modells vergleichen zu können, wird abermals eine exponentielle Glättung ohne die letzte Datenreihe durchgeführt und der extrahierte Wert prognostiziert. Anschließend kann mit der Differenz des tatsächlichen Werts zum prognostizierten Wert beurteilt werden, wie gut das Modell bei der Vorhersage abschneidet. In diesem Fall ergibt sich eine Abweichung in Höhe von: ', html.Strong(str(mae_links[1])), '.',
                ])
    if selected == 'movingAverage' and periods =='2':
        if x is not None:
            contents = x
            filename = y
            df = parse_contents(contents, filename)

            if df is None:
                return ''
            else:
                return html.P(children=[
                    'Dieser Graph zeigt die Ergebnisse des ',
                    html.Strong('Moving Average Modell der dritten Ordnung'),
                    ' . Das heißt basierend auf den jeweils drei vorherigen Datenpunkten wird der nächste Vorhergesagt.',
                    html.Br(), html.Br(), 'Um die Güte dieses Modells vergleichen zu können, wird abermals ein Moving-Average Verfahren ohne die letzte Datenreihe durchgeführt und der extrahierte Wert prognostiziert. Anschließend kann mit der Differenz des tatsächlichen Werts zum prognostizierten Wert beurteilt werden, wie gut das Modell bei der Vorhersage abschneidet. In diesem Fall ergibt sich eine Abweichung in Höhe von: ', html.Strong(str(mae_links[0])), '.',
                ])
  
    else:
        return ''

app.layout = layout

#Nun starte ich die App :)
if __name__ == '__main__':
    
    #app.server = server
    app.run_server(debug=True)
    
    
    