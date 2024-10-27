import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np









  

def get_radar_chart(input_data):
  
  input_data = get_scaled_values(input_data)
  
  categories = ['Radius', 'Perimeter', 'Area', 'Compactness','Concavity', 'Concave Points', 'Fractal Dimension']

  fig = go.Figure()

  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_mean'],input_data['perimeter_mean'],
          input_data['area_mean'], input_data['compactness_mean'],
          input_data['concavity_mean'], input_data['concave points_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
  ))
  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_se'], input_data['perimeter_se'], input_data['area_se']
          ],
        theta=categories,
        fill='toself',
        name='Standard Error'
  ))
  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_worst'], input_data['perimeter_worst'],
          input_data['area_worst'],  input_data['compactness_worst'],
          input_data['concavity_worst'], input_data['concave points_worst']
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

