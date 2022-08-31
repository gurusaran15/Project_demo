import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.pipeline import Pipeline


header = st.container()
dataset = st.container()
datamanipulation = st.container()
PCAalgo = st.container()
featureinput = st.container()
visualisations = st.container()
transfervisualisation = st.container()

#@st.cache
#def get_data(filename):
	#data=pd.read_csv(filename)
	#return data

with header:
	st.title('Welcome to my Sports Data Analytics Project')
	st.text('The following will be a my project demo with some sample visualisations') 


with dataset:
	st.header('I found this dataset in Kaggle.com which is the English Premier all-time Stats table until 2020')
	data = pd.read_csv('dataset - 2020-09-24.csv')
	st.write(data.head())
	st.header('This is a dataset I scrubbed off a scoreref that shows the Stats of all players in top 5 leagues for the 2021-2022 Season')
	data1 = pd.read_csv('file_name.csv')
	st.write(data1.head())
	transferdata= pd.read_csv('players.csv')
	transferdata=transferdata[transferdata['last_season']==2021]
	st.write(transferdata.head())

with datamanipulation:
	data['Tackle success %'] = data['Tackle success %'].str.rstrip('%').astype('float')/100.0
	data['Shooting accuracy %'] = data['Shooting accuracy %'].str.rstrip('%').astype('float')/100.0
	data['Cross accuracy %'] = data['Cross accuracy %'].str.rstrip('%').astype('float')/100.0
	data=data.replace(to_replace = np.nan, value =0)
	goalkeeper_data = data[data['Position'] == 'Goalkeeper']
	defender_data = data[data['Position'] == 'Defender']
	mid_data = data[data['Position'] == 'Midfielder']
	forward_data = data[data['Position'] == 'Forward']
	offense = pd.DataFrame(data, columns= ['Goals','Assists', 'Goals per match',
       'Headed goals', 'Goals with right foot', 'Goals with left foot',
       'Penalties scored', 'Freekicks scored', 'Shots', 'Shots on target',
       'Shooting accuracy %', 'Hit woodwork', 'Big chances missed'])
	defense = pd.DataFrame(data, columns= ['Clean sheets', 'Goals conceded', 'Tackles', 'Tackle success %',
       'Last man tackles', 'Blocked shots', 'Interceptions', 'Clearances',
       'Headed Clearance', 'Clearances off line', 'Recoveries', 'Duels won',
       'Successful 50/50s', 'Aerial battles won',])
	progression = pd.DataFrame(data, columns= ['Assists','Passes', 'Passes per match', 'Big chances created', 'Crosses',
       'Cross accuracy %', 'Through balls', 'Accurate long balls'])
	goalkeepin = pd.DataFrame(data, columns= ['Clean sheets','Saves','Penalties saved', 'Punches', 'High Claims', 'Catches',
       'Sweeper clearances', 'Throw outs', 'Goal Kicks'])
	errorinplay = pd.DataFrame(data, columns= ['Duels lost','Aerial battles lost','Yellow cards',
       'Red cards', 'Fouls', 'Offsides'])
	data_stats = data.loc[:, 'Wins':'Offsides'].columns.values
	attacktransfer = transferdata[transferdata['position']=='Attack']
	st.write(attacktransfer.head())
	defendertransfer = transferdata[transferdata['position']=='Defender']
	midfieldertransfer = transferdata[transferdata['position']=='Midfield']
	goalkeepertransfer = transferdata[transferdata['position']=='Goalkeeper']

#with datamanipulation1:
	#goalkeeper_data1 = data[data['Position']==]

with PCAalgo:
	st.header('We are gonna be using PCA algorithm on our dataset to determine the hidden gems in our dataset')
	pipe = Pipeline([('scaler', StandardScaler()),('decomposition', PCA(n_components=2))])
	off_transform=pipe.fit_transform(offense)
	def_transform=pipe.fit_transform(defense)
	prog_transform=pipe.fit_transform(progression)
	goalk_transform=pipe.fit_transform(goalkeepin)
	error_transform=pipe.fit_transform(errorinplay)
	
	final_off = pd.DataFrame(off_transform, columns=['Goal Potential', 'Assistive Potential'])
	#st.write(final_off.head())
	final_off = pd.merge(data[data['Position']=='Forward'], final_off, left_index=True, right_index=True)
	
	final_def = pd.DataFrame(def_transform, columns=['Defensive Experience', 'Tackles won'])
	#st.write(final_def.head())
	final_def = pd.merge(data[data['Position']=='Defender'], final_def, left_index=True, right_index=True)
	
	final_prog = pd.DataFrame(prog_transform, columns=['Passes Made', 'Progressive'])
	#st.write(final_prog.head())
	final_prog = pd.merge(data[data['Position']=='Midfielder'], final_prog, left_index=True, right_index=True)
	
	final_goalk = pd.DataFrame(goalk_transform, columns=['Saving Potential', 'Cleansheet Potential'])
	#st.write(final_goalk.head())
	final_goalk = pd.merge(data[data['Position']=='Goalkeeper'], final_goalk, left_index=True, right_index=True)
	
	final_error = pd.DataFrame(error_transform, columns=['Prone to fouls', 'Prone to Errors'])
	#st.write(final_error .head())
	final_error = pd.merge(data[data['Position']=='Defender'], final_error, left_index=True, right_index=True)

with featureinput:
	st.header('You can pick the different features below')
	sel_col,disp_col = st.columns(2)
	size = sel_col.slider('What will be the size of the visualisations', min_value=10, max_value=100, value=20, step=10)
	sel_col.text('Here is the list of features you can choose')
	input_feature = st.selectbox('Which feature do you want to visualize?',('Offensive','Defensive','Progression','Goalkeeper','Errors'))


with visualisations:
	if input_feature=='Offensive' :
		
	 fig = px.scatter(final_off, x="Goal Potential", y="Assistive Potential",
	          size="Goals", color="Club",
                  hover_name="Name", log_x=True, size_max=size)
	 st.plotly_chart(fig, use_container_width=False)
	pass
	if input_feature=='Defensive':
	 fig = px.scatter(final_def, x="Defensive Experience", y="Tackles won",
	          size="Tackles", color="Club",
                  hover_name="Name", log_x=True, size_max=size)
	 st.plotly_chart(fig, use_container_width=False)
	pass
	if input_feature=='Progression':
	 fig = px.scatter(final_prog, x="Passes Made", y="Progressive",
	          size="Big chances created", color="Club",
                  hover_name="Name", log_x=True, size_max=size)
	 st.plotly_chart(fig, use_container_width=False)
	pass
	if input_feature=='Goalkeeper':
	 fig = px.scatter(final_goalk, x="Saving Potential", y="Cleansheet Potential",
	          size="Clean sheets", color="Club",
                  hover_name="Name", log_x=True, size_max=size)
	 st.plotly_chart(fig, use_container_width=False)
	pass
	if input_feature=='Errors':
	 fig = px.scatter(final_error, x="Prone to fouls", y="Prone to Errors",
	          size="Yellow cards", color="Club",
                  hover_name="Name", log_x=True, size_max=size)
	 st.plotly_chart(fig, use_container_width=False)
	pass
	input_feature1 = st.selectbox('Which player transfer fee records do you wanna see?',('Attacker','Defender','Midfield','Goalkeeper'))

with transfervisualisation:

	if input_feature1=='Attacker':
		
	 fig = px.scatter(attacktransfer, x="Highest_value", y="market_value",
	          size="height_in_cm", color="country_of_citizenship",
                  hover_name="pretty_name", log_x=True, size_max=size)
	 st.plotly_chart(fig, use_container_width=False)
	pass 
	if input_feature1=='Defender':
		
	 fig = px.scatter(defendertransfer, x="Highest_value", y="market_value",
	          size="height_in_cm", color="country_of_citizenship",
                  hover_name="pretty_name", log_x=True, size_max=size)
	 st.plotly_chart(fig, use_container_width=False)
	pass 
	if input_feature1=='Midfield':
		
	 fig = px.scatter(midfieldertransfer, x="Highest_value", y="market_value",
	          size="height_in_cm", color="country_of_citizenship",
                  hover_name="pretty_name", log_x=True, size_max=size)
	 st.plotly_chart(fig, use_container_width=False)
	pass 
	if input_feature1=='Goalkeeper':
		
	 fig = px.scatter(goalkeepertransfer, x="Highest_value", y="market_value",
	          size="height_in_cm", color="country_of_citizenship",
                  hover_name="pretty_name", log_x=True, size_max=size)
	 st.plotly_chart(fig, use_container_width=False)
	pass 











