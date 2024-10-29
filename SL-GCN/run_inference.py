import torch
from model.decouple_gcn_attn import Model  # Replace with actual model class
from torchvision import transforms
from PIL import Image
import yaml
import numpy as np

# Load the pre-trained model
def load_model(weights_path, config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Access the nested 'model_args' dictionary in the YAML file
    model_args = config['model_args']

    # Get the necessary parameters from 'model_args'
    num_class = model_args['num_class']
    num_point = model_args['num_point']
    num_person = model_args['num_person']
    graph = model_args['graph']
    groups = model_args['groups']
    block_size = model_args['block_size']
    graph_args = model_args['graph_args']
    in_channels = model_args.get('in_channels', 3)  # Default to 3 if not in YAML

    # Initialize the model with the retrieved parameters
    model = Model(num_class=num_class, num_point=num_point, 
                  num_person=num_person, groups=groups, 
                  block_size=block_size, graph=graph, 
                  graph_args=graph_args, in_channels=in_channels)

    # Force the model to run on the cpu
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.to(torch.device('cpu'))  # Ensure the model is on cpu
    model.eval()  # Set the model to evaluation mode
    return model

# Preprocess the input image
def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512)),  # Resize to 512x512
                  # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize as needed
    ])
    #image = Image.open(image_path)
    image = np.load(image_path, allow_pickle=True)
    input_tensor = preprocess(image).unsqueeze(0).unsqueeze(2)  # Add batch dimension
    input_tensor = input_tensor.view((1, 1, 512, 512, 3))
    print("tensor shape", input_tensor.shape)
    return input_tensor.to(torch.device('cpu'))

# Run inference
def run_inference(model, input_tensor):
    # Move input to cpu
    input_tensor = input_tensor.to(torch.device('cpu'))  # Ensure input is on cpu
    with torch.no_grad():
        output = model(input_tensor)
    return output

if __name__ == "__main__":
    weights_path = "/Users/siduppuluri/Desktop/CSE 485/SAM-SLR-v2/SL-GCN/pretrained_models/autsl_bone_epoch_239.pt"  # Path to your .pt file
    config_path = "/Users/siduppuluri/Desktop/CSE 485/SAM-SLR-v2/SL-GCN/config/AUTSL/test/test_bone.yaml"    # Adjust this to the right configuration file
    #image_path = "/Users/siduppuluri/Desktop/CSE 485/SAM-SLR-v2/SL-GCN/input_images/testingmodel.jpg"                # Path to your input image

    image_path = '/Users/siduppuluri/Downloads/test_data_joint_motion.npy'
    # Load and preprocess the image
    #input_tensor = preprocess_image(image_path)
    input_tensor = np.load(image_path)
    input_tensor = torch.tensor(input_tensor).to("cpu")
    print(input_tensor.shape)

    # Load the model
    model = load_model(weights_path, config_path).to('cpu')
    
    # Run inference
    output = run_inference(model, input_tensor)
    print("Inference Output:", output)