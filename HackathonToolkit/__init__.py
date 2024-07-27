import os
import subprocess
import sys
from flask import Flask, jsonify
import threading
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import random
import string
import torch
import tensorflow as tf
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import torchvision.models as tv_models
import math
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import git
import boto3
import subprocess
from google.cloud import container_v1
from google.oauth2 import service_account
import time
import requests
from bs4 import BeautifulSoup
import shutil
from datetime import datetime
import platform
import subprocess




def greeting():
    print('Hello! Welcome to the Hackathon Toolkit!')


def create_project_structure(project_name: str):
    dirs = [
        os.path.join(project_name, "src"),
        os.path.join(project_name, "tests"),
        os.path.join(project_name, "docs")
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print(f"Project structure for '{project_name}' created successfully.")
    
    
def add_readme(project_name: str, content: str):
    filename = "README.md"
    with open(filename, "w") as file:
        file.write(f"# {project_name}\n\n")
        file.write(content)
    print(f"{filename} has been created with the provided content.")
    
    
def create_virtual_environment(project_path: str, python_version: str):
    try:
        venv_path = f"{project_path}/venv"
        subprocess.check_call([sys.executable, "-m", "venv", venv_path])
        activate_script = f"{venv_path}/bin/activate"
        print(f"Virtual environment created. To activate, run:\nsource {activate_script}")
        if python_version:
            subprocess.check_call([f"{venv_path}/bin/pip", "install", f"python=={python_version}"])
            print(f"Python {python_version} has been installed in the virtual environment.")
        else:
            print("No specific Python version provided, using default version.")
    except Exception as e:
        print(f"An error occurred: {e}")
        
        
def install_dependencies(requirements_file: str):
    try:
        with open(requirements_file, "r") as file:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        print(f"Dependencies from {requirements_file} have been installed successfully.")
    except FileNotFoundError:
        print(f"Error: {requirements_file} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        

def generate_api_wrapper(api_name: str, endpoints: dict):
    class_name = api_name.capitalize() + "API"
    wrapper_code = f"class {class_name}:\n"
    wrapper_code += "    def __init__(self, base_url, headers=None):\n"
    wrapper_code += "        self.base_url = base_url\n"
    wrapper_code += "        self.headers = headers if headers else {}\n\n"
    for endpoint_name, endpoint_details in endpoints.items():
        method_name = endpoint_name.lower()
        http_method = endpoint_details.get("method", "GET").upper()
        path = endpoint_details.get("path", "")
        params = endpoint_details.get("params", [])
        param_str = ", ".join(params)
        wrapper_code += f"    def {method_name}(self, {param_str}):\n"
        wrapper_code += f"        url = f\"{{self.base_url}}{path}\"\n"
        wrapper_code += f"        response = requests.{http_method.lower()}(url, headers=self.headers"
        if http_method in ["POST", "PUT", "PATCH"]:
            wrapper_code += ", json={param_str}"
        wrapper_code += ")\n"
        wrapper_code += "        return response.json()\n\n"
    return wrapper_code


def start_mock_server(mock_data: dict, port: int):
    app = Flask(__name__)
    for endpoint, data in mock_data.items():
        def generate_serve_function(data):
            def serve_mock_data():
                return jsonify(data)
            return serve_mock_data
        serve_function = generate_serve_function(data)
        endpoint_name = f"serve_mock_data_{endpoint.replace('/', '_')}"
        app.add_url_rule(endpoint, endpoint_name, serve_function, methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    def run_server():
        app.run(port=port)
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()
    print(f"Mock server started on port {port}")
    
    
def load_csv(file_path: str):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error: Failed to load CSV file '{file_path}'. Reason: {str(e)}")
        return None


def load_json(file_path: str):
    try:
        df = pd.read_json(file_path)
        return df
    except Exception as e:
        print(f"Error loading JSON file: {str(e)}")
        return None


def augment_image(image_path: str, augmentation_type: str, save_path: str):
    try:
        image = Image.open(image_path)
        
        if augmentation_type == 'rotate':
            # Rotate the image by a random angle
            angle = random.randint(-30, 30)  # Random angle between -30 and 30 degrees
            image = image.rotate(angle)
        elif augmentation_type == 'flip':
            # Flip the image horizontally or vertically
            flip_type = random.choice(['horizontal', 'vertical'])
            if flip_type == 'horizontal':
                image = ImageOps.mirror(image)
            elif flip_type == 'vertical':
                image = ImageOps.flip(image)
        elif augmentation_type == 'brightness':
            # Adjust brightness
            enhancer = ImageEnhance.Brightness(image)
            factor = random.uniform(0.5, 1.5)  # Random factor between 0.5 and 1.5
            image = enhancer.enhance(factor)
        elif augmentation_type == 'contrast':
            # Adjust contrast
            enhancer = ImageEnhance.Contrast(image)
            factor = random.uniform(0.5, 1.5)  # Random factor between 0.5 and 1.5
            image = enhancer.enhance(factor)
        elif augmentation_type == 'blur':
            # Apply Gaussian blur
            radius = random.randint(1, 2)  # Random blur radius
            image = image.filter(ImageFilter.GaussianBlur(radius))
        else:
            print(f"Unsupported augmentation type: {augmentation_type}")
            return None
        
        # Save the augmented image
        image.save(save_path)
        print(f"Augmented image saved to {save_path}")
        
        return image
    
    except Exception as e:
        print(f"Error applying augmentation: {str(e)}")
        return None


def load_pretrained_models(model_names):
    pretrained_models = {}
    
    for model_name in model_names:
        if model_name.startswith("tf-"):
            # TensorFlow model
            tf_model_name = model_name[3:]  # Remove "tf-" prefix
            if tf_model_name == "resnet50":
                model = ResNet50(weights="imagenet")
                pretrained_models[model_name] = model
            else:
                raise ValueError(f"Unknown TensorFlow model: {tf_model_name}")
        
        elif model_name.startswith("pt-"):
            # PyTorch model
            pt_model_name = model_name[3:]  # Remove "pt-" prefix
            if pt_model_name == "bert-base-uncased":
                model = AutoModelForSequenceClassification.from_pretrained(pt_model_name)
                tokenizer = AutoTokenizer.from_pretrained(pt_model_name)
                pretrained_models[model_name] = (model, tokenizer)
            else:
                raise ValueError(f"Unknown PyTorch model: {pt_model_name}")
        
        else:
            raise ValueError(f"Unknown model prefix: {model_name}")
    
    return pretrained_models


def list_available_models():
    available_models = dir(tv_models)
    return available_models


def calculate_accuracy(y_true, y_pred):
    correct = 0
    total = len(y_true)
    
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def calculate_rmse(y_true, y_pred):

    if len(y_true) != len(y_pred):
        raise ValueError("Lengths of y_true and y_pred must be the same.")
    
    # Calculate RMSE
    squared_errors = [(yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)]
    mean_squared_error = sum(squared_errors) / len(y_true)
    rmse = math.sqrt(mean_squared_error)
    
    return rmse


def grid_search(model, param_grid, X_train, y_train):
   
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    return grid_search


def random_search(model, param_distributions, X_train, y_train, n_iter=10):
    param_grid = {key: (list(range(val[0], val[1] + 1)) if isinstance(val, tuple) else val)
                  for key, val in param_distributions.items()}
    
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid,
                                       n_iter=n_iter, cv=5, scoring='accuracy', random_state=42)
    random_search.fit(X_train, y_train)
    
    return random_search


def plot_confusion_matrix(y_true, y_pred, classes):
 
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


def plot_roc_curve(y_true, y_pred_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()


def generate_dashboard(data, output_file):
 
    # Convert data dictionary to a DataFrame for easier manipulation
    df = pd.DataFrame(data)

    # Plotting metrics
    plt.figure(figsize=(12, 8))

    # Example: Plotting metric1 as a line plot
    plt.subplot(2, 2, 1)
    sns.lineplot(data=df['metric1'], marker='o')
    plt.title('Metric 1')

    # Example: Plotting metric2 as a bar plot
    plt.subplot(2, 2, 2)
    sns.barplot(x=df.index, y=df['metric2'])
    plt.title('Metric 2')

    # Example: Plotting metric3 as a scatter plot
    plt.subplot(2, 2, 3)
    sns.scatterplot(x=df['metric3'], y=df['metric4'], hue=df['metric5'])
    plt.title('Metric 3 vs Metric 4')

    # Save the dashboard to the specified output file
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()


def git_init(project_path: str):
    if not os.path.exists(project_path):
        os.makedirs(project_path)
    
    try:
        repo = git.Repo.init(project_path)
        return repo
    except git.exc.GitError as e:
        raise RuntimeError(f"An error occurred while initializing the Git repository: {e}")


def git_commit_all(message: str):
    # Check if current directory is a Git repository, otherwise initialize one
    repo = git.Repo(os.getcwd())
    if repo.bare:
        raise git.InvalidGitRepositoryError("Not a valid Git repository")

    # Add all changes
    repo.git.add('--all')

    # Commit changes
    commit = repo.index.commit(message)
    return commit
    
    
def git_push(remote: str, branch: str):
    repo = git.Repo(os.getcwd())
    if repo.bare:
        raise git.InvalidGitRepositoryError("Not a valid Git repository")

    repo.remotes[remote].push(branch)
    
    
def deploy_to_aws(project_path: str, aws_credentials: dict):
    # Extract AWS credentials from input dictionary
    access_key = aws_credentials.get('access_key')
    secret_key = aws_credentials.get('secret_key')
    region = aws_credentials.get('region', 'us-east-1')  # Default region if not provided

    # Set AWS credentials for Boto3
    os.environ['AWS_ACCESS_KEY_ID'] = access_key
    os.environ['AWS_SECRET_ACCESS_KEY'] = secret_key
    os.environ['AWS_DEFAULT_REGION'] = region

    # Example: Build Docker image
    build_command = f'docker build -t myapp {project_path}'
    subprocess.run(build_command, shell=True, check=True)

    # Example: Push Docker image to AWS ECR (Elastic Container Registry)
    ecr_client = boto3.client('ecr')
    response = ecr_client.get_authorization_token()
    token = response['authorizationData'][0]['authorizationToken']
    registry = response['authorizationData'][0]['proxyEndpoint']

    subprocess.run(f'docker login -u AWS -p {token} {registry}', shell=True, check=True)
    subprocess.run(f'docker tag myapp:latest {registry}/myapp:latest', shell=True, check=True)
    subprocess.run(f'docker push {registry}/myapp:latest', shell=True, check=True)

    # Example: Deploy Docker image to AWS ECS (Elastic Container Service)
    ecs_client = boto3.client('ecs')

    # Create ECS task definition
    task_definition = {
        'family': 'myapp-task',
        'containerDefinitions': [
            {
                'name': 'myapp-container',
                'image': f'{registry}/myapp:latest',
                'cpu': 256,
                'memory': 512,
                'portMappings': [
                    {
                        'containerPort': 80,
                        'hostPort': 80
                    }
                ]
            }
        ],
        'requiresCompatibilities': [
            'FARGATE'
        ],
        'networkMode': 'awsvpc'
    }

    response = ecs_client.register_task_definition(**task_definition)
    task_definition_arn = response['taskDefinition']['taskDefinitionArn']

    # Run ECS service with the task definition
    service_definition = {
        'cluster': 'myapp-cluster',
        'serviceName': 'myapp-service',
        'taskDefinition': task_definition_arn,
        'desiredCount': 1,
        'launchType': 'FARGATE',
        'networkConfiguration': {
            'awsvpcConfiguration': {
                'subnets': ['subnet-12345678'],
                'securityGroups': ['sg-87654321'],
                'assignPublicIp': 'ENABLED'
            }
        }
    }

    ecs_client.create_service(**service_definition)

    print("Deployment completed successfully.")


def deploy_to_gcp(project_path: str, gcp_credentials: dict):
    # Extract GCP credentials from input dictionary
    credentials_path = gcp_credentials.get('credentials_path')
    project_id = gcp_credentials.get('project_id')
    region = gcp_credentials.get('region', 'us-central1')  # Default region if not provided

    # Set environment variable for Google Application Credentials
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

    # Example: Build Docker image
    build_command = f'docker build -t myapp {project_path}'
    subprocess.run(build_command, shell=True, check=True)

    # Example: Push Docker image to Google Container Registry (GCR)
    gcr_image = f'gcr.io/{project_id}/myapp:latest'
    push_command = f'docker tag myapp:latest {gcr_image}'
    subprocess.run(push_command, shell=True, check=True)
    subprocess.run(f'docker push {gcr_image}', shell=True, check=True)

    # Example: Deploy Docker image to Google Kubernetes Engine (GKE)
    client = container_v1.ClusterManagerClient()

    # Replace with your GKE cluster and namespace details
    cluster_id = 'your-cluster-id'
    namespace = 'default'

    # Get credentials for the cluster
    cluster = client.get_cluster(project_id, region, cluster_id)
    cluster_endpoint = cluster.endpoint
    cluster_ca_certificate = cluster.master_auth.cluster_ca_certificate

    # Example: Create deployment
    deployment_body = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": "myapp-deployment",
            "labels": {
                "app": "myapp"
            }
        },
        "spec": {
            "replicas": 1,
            "selector": {
                "matchLabels": {
                    "app": "myapp"
                }
            },
            "template": {
                "metadata": {
                    "labels": {
                        "app": "myapp"
                    }
                },
                "spec": {
                    "containers": [
                        {
                            "name": "myapp",
                            "image": gcr_image,
                            "ports": [
                                {
                                    "containerPort": 8080
                                }
                            ]
                        }
                    ]
                }
            }
        }
    }

    # Deploy to GKE
    api = client.api
    api.create_namespaced_deployment(namespace, deployment_body)

    print("Deployment completed successfully.")


timers = {}

def start_timer(task_name: str):
    if task_name in timers:
        print(f"Timer for '{task_name}' is already running.")
    else:
        timers[task_name] = time.time()
        print(f"Timer started for '{task_name}'.")

def stop_timer(task_name: str):
    if task_name in timers:
        elapsed_time = time.time() - timers[task_name]
        del timers[task_name]
        print(f"Timer stopped for '{task_name}'. Elapsed time: {elapsed_time:.2f} seconds.")
        return elapsed_time
    else:
        print(f"No timer found for '{task_name}'.")
        return None


def web_scrape(url: str, element: str):
    try:
        # Fetch the webpage content
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad responses

        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find elements matching the specified tag or class
        elements = soup.find_all(element)

        # Extract and print the text of each element found
        for elem in elements:
            print(elem.text.strip())

    except requests.exceptions.RequestException as e:
        print(f"Error fetching webpage: {e}")
    except Exception as e:
        print(f"Error: {e}")


def backup_data(source_dir, backup_dir):

    try:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        backup_folder = os.path.join(backup_dir, f'backup_{timestamp}')
        
        os.makedirs(backup_folder, exist_ok=True)
        
        for item in os.listdir(source_dir):
            source_item = os.path.join(source_dir, item)
            if os.path.isfile(source_item):
                shutil.copy2(source_item, backup_folder)
            elif os.path.isdir(source_item):
                shutil.copytree(source_item, os.path.join(backup_folder, item))
        
        print(f"Backup successful. Files saved in {backup_folder}")
        return True
    except Exception as e:
        print(f"Backup failed: {str(e)}")
        return False


def check_os_compatibility():
    os_name = platform.system()
    compatibility_status = "Compatible"
    
    if os_name == "Windows":
        compatibility_status = "Compatible"
        # Add Windows-specific compatibility checks here
    elif os_name == "Linux":
        compatibility_status = "Compatible"
        # Add Linux-specific compatibility checks here
    elif os_name == "Darwin":
        compatibility_status = "Compatible"
        # Add macOS-specific compatibility checks here
    else:
        compatibility_status = "Unknown OS"
    
    return f"Operating System: {os_name}, Compatibility Status: {compatibility_status}"


def run_cross_platform_tests():
    
    test_results = {
        "OS Detection": None,
        "File System Access": None,
        "Basic Command Execution": None
    }
    
    # OS Detection Test
    try:
        os_name = platform.system()
        test_results["OS Detection"] = f"Detected OS: {os_name}"
    except Exception as e:
        test_results["OS Detection"] = f"Failed to detect OS: {str(e)}"
    
    # File System Access Test
    try:
        test_file = "test_file.txt"
        with open(test_file, 'w') as file:
            file.write("This is a test file.")
        with open(test_file, 'r') as file:
            content = file.read()
        os.remove(test_file)
        test_results["File System Access"] = "File system access successful"
    except Exception as e:
        test_results["File System Access"] = f"File system access failed: {str(e)}"
    
    # Basic Command Execution Test
    try:
        if os_name == "Windows":
            result = subprocess.run(["echo", "Hello, World!"], capture_output=True, text=True)
        else:
            result = subprocess.run(["echo", "Hello, World!"], capture_output=True, text=True)
        if result.returncode == 0:
            test_results["Basic Command Execution"] = f"Command executed successfully: {result.stdout.strip()}"
        else:
            test_results["Basic Command Execution"] = f"Command execution failed with return code {result.returncode}"
    except Exception as e:
        test_results["Basic Command Execution"] = f"Command execution failed: {str(e)}"
    
    return test_results
