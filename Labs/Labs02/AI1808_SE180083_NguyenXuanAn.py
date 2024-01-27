import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('Labs\\Labs02\\diabetes_binary_health_indicators_BRFSS2015.csv', encoding="ISO-8859-1")

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Diabetes Health"),

    dcc.Dropdown(
        id='classifier-dropdown',
        options=[
            {'label': 'Logistic Regression', 'value': 'logistic'},
            {'label': 'KNN', 'value': 'KNN'},
            {'label': 'RandomForest Classifier', 'value': 'randomforest'}
        ],
        value='logistic'
    ),
    html.Div(id='output'),
    html.Br(),
    html.H1("Diabetes Health Visualization"),
    html.Br(),
    html.Div([
        html.Div([
            dcc.Graph(id='heatmap', style={'width': '100%', 'height': '100%'}),
        ], style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center'}),

        html.Div([
            dcc.Graph(id='distribution-plot', style={'width': '100%', 'height': '100%'}),
        ], style={'margin': '20px'})
    ], style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center'}),

    html.Div([
            dcc.Graph(id='box-plot', style={'width': '100%', 'height': '100%', 'display': 'flex', 'justify-content': 'center', 'align-items': 'center'})
    ])
])


@app.callback(
    Output('heatmap', 'figure'),
    [Input('classifier-dropdown', 'value')]
)
def update_heatmap(selected_variable):
    correlation_matrix = df.corr()

    fig = px.imshow(correlation_matrix,
                    labels=dict(color="Correlation"),
                    x=correlation_matrix.columns,
                    y=correlation_matrix.columns,
                    color_continuous_scale='Viridis')
    fig.update_layout(title='Correlation Matrix Heatmap',
                    height=500, width=500)
    return fig


@app.callback(
    Output('distribution-plot', 'figure'),
    [Input('classifier-dropdown', 'value')]
)
def update_distribution_plot(selected_variable):
    fig = px.histogram(df, x='BMI')
    fig.update_layout(title='BMI Histogram',
                    height=500, width=500)
    return fig

@app.callback(
    Output('box-plot', 'figure'),
    [Input('classifier-dropdown', 'value')]
)
def update_box_plot(selected_variable):
    fig = px.box(df, y='BMI')
    fig.update_layout(title='BMI Boxplot',
                    height=500, width=500)
    return fig

@app.callback(
    Output('output', 'children'),
    [Input('classifier-dropdown', 'value')]
    )
def generate_results(selected_classifier):
    if selected_classifier == 'KNN':
        model = KNeighborsClassifier(n_neighbors=3)
    elif selected_classifier == 'logistic':
        model = LogisticRegression()
    elif selected_classifier == 'randomforest':
        model = RandomForestClassifier()

    X = df.drop(columns=['Diabetes_binary'])
    y = df['Diabetes_binary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])
    fig = px.imshow(cm_df,
                    labels=dict(x="Predicted", y="Actual"),
                    color_continuous_scale='Viridis',
                    title="Confusion Matrix",
                    zmin=0, zmax=cm.max().max(),
                    width=1000, height=800
                    )
    fig.update_layout(coloraxis_colorbar=dict(title="Count"))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            fig.add_annotation(x=j, y=i, text=str(cm[i, j]), showarrow=False, font=dict(color='white', size=14))
    return html.Div([
        html.H4(f"Accuracy: {accuracy:.2f}", style={'font-size': '24px', 'text-align': 'center'}),
        dcc.Graph(figure=fig)
    ])

if __name__ == '__main__':
    app.run_server(debug=True)