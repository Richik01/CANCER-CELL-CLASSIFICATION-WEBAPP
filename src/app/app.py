import os
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import pickle
import torch
import torch.nn as nn
import streamlit as st
# LOADING THE MODEL AND SCALER
class LogisticRegression(nn.Module):
    def __init__(self, n):
        super(LogisticRegression, self).__init__()
        self.input_layer = nn.Linear(n, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out = self.input_layer(x)
        act_out = self.sig(out)
        return act_out

model = LogisticRegression(15)
model.load_state_dict(torch.load('data/model.pth', weights_only=True))

with open('data/scaler.pkl', 'rb') as s:
    ss = pickle.load(s)

def get_cleaned_data():
   data = pd.read_csv('data/cleaned_data.csv')
   return data
def binary_fun(x):
    return 0 if x<0.5 else 1 

#UI
def add_sidebar():
  st.sidebar.header("Cell Nuclei Measurements")
  
  data = get_cleaned_data()
  
  slider_labels = [
        ("Radius (mean)", "radius1"),
        ("Perimeter (mean)", "perimeter1"),
        ("Area (mean)", "area1"),
        ("Compactness (mean)", "compactness1"),
        ("Concavity (mean)", "concavity1"),
        ("Concave points (mean)", "concave_points1"),
        ("Radius (se)", "radius2"),
        ("Perimeter (se)", "perimeter2"),
        ("Area (se)", "area2"),
        ("Radius (worst)", "radius3"),
        ("Perimeter (worst)", "perimeter3"),
        ("Area (worst)", "area3"),
        ("Compactness (worst)", "compactness3"),
        ("Concavity (worst)", "concavity3"),
        ("Concave points (worst)", "concave_points3"),
    ]

  input_dict = {}

  for label, key in slider_labels:
    input_dict[key] = st.sidebar.slider(
      label,
      min_value=float(0),
      max_value=float(data[key].max()),
      value=float(data[key].mean())
    )
    
  return input_dict

def get_scaled_values(input_dict):
  print(list(input_dict.values()))
  foo = np.array(list(input_dict.values()))
  foo = foo.reshape(1, -1)
  print(foo)
  scaled_foo = ss.transform(foo)
  scaled_dict = {}
  i=0
  for key, value in input_dict.items():
    scaled_dict[key] = scaled_foo[0][i]
    i+=1
  print(scaled_dict)
  return scaled_dict

def get_radar_chart(input_data):
  
  input_data = get_scaled_values(input_data)
  
  categories = ['Radius', 'Perimeter', 'Area', 'Compactness','Concavity', 'Concave Points', 'Fractal Dimension']

  fig = go.Figure()

  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius1'],input_data['perimeter1'],
          input_data['area1'], input_data['compactness1'],
          input_data['concavity1'], input_data['concave_points1']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
  ))
  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius2'], input_data['perimeter2'], input_data['area2']
          ],
        theta=categories,
        fill='toself',
        name='Standard Error'
  ))
  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius3'], input_data['perimeter3'],
          input_data['area3'],  input_data['compactness3'],
          input_data['concavity3'], input_data['concave_points3']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
  ))

  fig.update_layout(
    polar=dict(
      radialaxis=dict(
        visible=True,
        range=[0, 1]
      )),
    showlegend=True
  )
  
  return fig


def add_predictions(input_data):
  input_array = np.array(list(input_data.values())).reshape(1, -1)
  
  input_array_scaled = ss.transform(input_array)
  
  model_input_array = torch.tensor(input_array_scaled, dtype=torch.float32)
  prediction = model(model_input_array)
  
  st.subheader("Cell cluster prediction")
  st.write("The cell cluster is:")
  
  if binary_fun(prediction) == 0:
    st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
  else:
    st.write("<span class='diagnosis malicious'>Malicious</span>", unsafe_allow_html=True)
  
  
  if binary_fun(prediction) == 0:
    st.write("Probability of being benign: {:.3f}".format(float(prediction.detach()[0][0])))
    st.write("Probability of being malicious: {:.3f}".format(1 - float(prediction.detach()[0][0])))
  else:
    st.write("Probability of being malicious: ", prediction)
    st.write("Probability of being benign: ", 1 - prediction)
     
  st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")



def main():
  st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon=":female-doctor:",
    layout="wide",
    initial_sidebar_state="expanded"
  )
  
  with open("assets/style.css") as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
  
  input_data = add_sidebar()
  
  with st.container():
    st.title("Breast Cancer Predictor")
  
  col1, col2 = st.columns([4,1])
  
  with col1:
    radar_chart = get_radar_chart(input_data)
    st.plotly_chart(radar_chart)
  with col2:
    add_predictions(input_data)


 
if __name__ == '__main__':
  main()