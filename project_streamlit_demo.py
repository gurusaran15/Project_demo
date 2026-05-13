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


def docker_dev():
    """Start development environment — mirrors `make docker_dev`."""
    storage_type = os.environ.get("SNAPSHOT_STORAGE_TYPE", "local")
    print(f"Starting development environment with {storage_type} storage...")
    print("Cleaning up any existing containers...")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dev_tools = os.path.join(os.getcwd(), "dev-tools")
    if storage_type == "s3":
        s3_bucket = os.environ.get("SNAPSHOT_S3_BUCKET")
        if not s3_bucket:
            print("ERROR: SNAPSHOT_S3_BUCKET environment variable required for S3 storage")
            sys.exit(1)
        aws_creds = subprocess.run(
            ["aws", "configure", "export-credentials", "--profile", os.environ.get("AWS_PROFILE", ""), "--format", "env"],
            capture_output=True, text=True
        )
        env = {**os.environ, **dict(line.split("=", 1) for line in aws_creds.stdout.splitlines() if "=" in line)}
        env.update({
            "DEPLOYMENT_ID": commit,
            "SNAPSHOT_STORAGE_TYPE": "s3",
            "SNAPSHOT_S3_BUCKET": s3_bucket,
            "SNAPSHOT_S3_PREFIX": os.environ.get("SNAPSHOT_S3_PREFIX", "snapshot"),
        })
        subprocess.run(["docker-compose", "down", "--remove-orphans"], cwd=dev_tools, env=env)
        print("Starting development environment with S3 snapshot...")
        result = subprocess.run(["docker-compose", "up"], cwd=dev_tools, env=env)
    else:
        env = {**os.environ,
            "DEPLOYMENT_ID": commit,
            "SNAPSHOT_STORAGE_TYPE": "local",
            "SNAPSHOT_STORAGE_PATH": "../dev_cache_storage",
        }
        subprocess.run(["docker-compose", "down", "--remove-orphans"], cwd=dev_tools, env=env)
        print("Starting development environment with local snapshot...")
        result = subprocess.run(["docker-compose", "up"], cwd=dev_tools, env=env)
    sys.exit(result.returncode)


def docker_dev_full_stack():
    """Build fresh snapshot then start Docker — mirrors `make docker_dev_full_stack`."""
    storage_type = os.environ.get("SNAPSHOT_STORAGE_TYPE", "local")
    print(f"Starting batteries included development environment with {storage_type} storage...")
    print("Step 1: Building fresh snapshot from Athena...")
    snapshot_build()
    print("Step 2: Starting Docker development environment...")
    docker_dev()


def docker_dev_from_zip():
    """Start development environment from a zip file — mirrors `make docker_dev_from_zip`."""
    zipfile = os.environ.get("ZIPFILE")
    if not zipfile:
        print("ERROR: ZIPFILE parameter required. Usage: ZIPFILE=path/to/snapshot.zip python scripts/cli.py docker_dev_from_zip")
        sys.exit(1)
    if not os.path.isfile(zipfile):
        print(f"ERROR: Zip file {zipfile} does not exist")
        sys.exit(1)
    print("Setting up development environment from downloaded snapshot zip...")
    print(f"Using zip file: {zipfile}")
    snapshot_clean()
    print("Copying snapshot zip to storage directory...")
    os.makedirs("dev_cache_storage", exist_ok=True)
    subprocess.run(["cp", zipfile, "dev_cache_storage/downloaded_snapshot.zip"])
    print("Snapshot setup complete")
    dev_tools = os.path.join(os.getcwd(), "dev-tools")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    env = {**os.environ,
        "DEPLOYMENT_ID": commit,
        "SNAPSHOT_STORAGE_TYPE": "local",
        "SNAPSHOT_STORAGE_PATH": "../dev_cache_storage",
        "SNAPSHOT_ALLOW_FALLBACK": "true",
    }
    print("Cleaning up existing containers...")
    subprocess.run(["docker-compose", "down", "--remove-orphans"], cwd=dev_tools, env=env)
    print("Starting development environment with fallback mode...")
    result = subprocess.run(["docker-compose", "up"], cwd=dev_tools, env=env)
    sys.exit(result.returncode)


def docker_stop():
    """Stop Docker development environment — mirrors `make docker_stop`."""
    print("Stopping Docker development environment...")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dev_tools = os.path.join(os.getcwd(), "dev-tools")
    env = {**os.environ,
        "DEPLOYMENT_ID": commit,
        "SNAPSHOT_STORAGE_TYPE": "local",
        "SNAPSHOT_STORAGE_PATH": "../dev_cache_storage",
    }
    result = subprocess.run(["docker-compose", "down", "--remove-orphans", "--volumes"], cwd=dev_tools, env=env)
    sys.exit(result.returncode)


def docker_clean():
    """Force clean all Docker containers and networks — mirrors `make docker_clean`."""
    print("Force cleaning all Docker containers and networks...")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dev_tools = os.path.join(os.getcwd(), "dev-tools")
    env = {**os.environ,
        "DEPLOYMENT_ID": commit,
        "SNAPSHOT_STORAGE_TYPE": "local",
        "SNAPSHOT_STORAGE_PATH": "../dev_cache_storage",
    }
    subprocess.run(["docker-compose", "down", "--remove-orphans", "--volumes"], cwd=dev_tools, env=env)
    subprocess.run(["docker", "system", "prune", "-f", "--filter", "label=com.docker.compose.project=dev-tools"])
    print("Docker cleanup complete")


def deploy_fallback_emails():
    """Copy fallback emails file to pod — mirrors `make deploy_fallback_emails`."""
    pod = os.environ.get("POD")
    if not pod:
        print("ERROR: POD parameter required. Usage: POD=your-pod-name python scripts/cli.py deploy_fallback_emails")
        sys.exit(1)
    print(f"Copying fallback emails file to pod {pod}...")
    result = subprocess.run(
        ["kubectl", "cp", "backend/config/fallback_emails.txt", f"{pod}:/app/config/fallback_emails.txt"]
    )
    print("File deployed successfully!")
    sys.exit(result.returncode)


def copy_fallback_emails():
    """Copy fallback emails file from pod to local — mirrors `make copy_fallback_emails`."""
    pod = os.environ.get("POD")
    if not pod:
        print("ERROR: POD parameter required. Usage: POD=your-pod-name python scripts/cli.py copy_fallback_emails")
        sys.exit(1)
    print(f"Copying fallback emails file from pod {pod}...")
    subprocess.run(
        ["kubectl", "cp", f"{pod}:/app/config/fallback_emails.txt", "backend/config/fallback_emails.txt"]
    )
    print("File copied to backend/config/fallback_emails.txt")
    print("Current contents:")
    result = subprocess.run(["cat", "backend/config/fallback_emails.txt"])
    sys.exit(result.returncode)


def k8s_latest_snapshot_logs():
    """View logs from latest cache build job — mirrors `make k8s_latest_snapshot_logs`."""
    print("Finding cache build job in current namespace...")
    job = subprocess.run(
        ["kubectl", "get", "jobs", "--sort-by=.metadata.creationTimestamp", "-o", "name"],
        capture_output=True, text=True
    )
    jobs = [j for j in job.stdout.splitlines() if "cache-build-" in j]
    if not jobs:
        print("No cache build jobs found in current namespace")
        print("Available jobs:")
        subprocess.run(["kubectl", "get", "jobs"])
        sys.exit(1)
    job_name = jobs[-1].split("/")[-1]
    print(f"Found job: {job_name}")
    pod = subprocess.run(
        ["kubectl", "get", "pods", f"--selector=job-name={job_name}", "-o", "jsonpath={.items[0].metadata.name}"],
        capture_output=True, text=True
    )
    pod_name = pod.stdout.strip()
    if not pod_name:
        print(f"No pod found for job {job_name}")
        subprocess.run(["kubectl", "describe", "job", job_name])
        sys.exit(1)
    print(f"Found pod: {pod_name}")
    pod_status = subprocess.run(
        ["kubectl", "get", "pod", pod_name, "-o", "jsonpath={.status.phase}"],
        capture_output=True, text=True
    ).stdout.strip()
    print(f"Pod status: {pod_status}")
    if pod_status == "Running":
        print("Following live logs (Ctrl+C to exit):")
        result = subprocess.run(["kubectl", "logs", "-f", pod_name])
    else:
        print("Showing logs from completed job:")
        result = subprocess.run(["kubectl", "logs", pod_name])
    sys.exit(result.returncode)










