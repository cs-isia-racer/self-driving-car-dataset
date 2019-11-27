import argparse
import json

from PIL import Image

from save import svr_from_config
from datautil import Pipeline, stripped_resnet18

def get_config():
    parser = argparse.ArgumentParser(description='Predict the steering angle for a given image')
    parser.add_argument('params', type=str, help='path to the SVR parameters')
    parser.add_argument('img', type=str, help='path to the image to analyze')
    
    return parser.parse_args()

def rescale(angles):
    return [270 * angle - 135 for angle in angles]
    
def main():
    conf = get_config()
    
    with open(conf.params, "r") as f:
        model = svr_from_config(json.load(f))
        
    p = Pipeline(stripped_resnet18(), model)
    
    images = [Image.open(conf.img)]
        
    print(rescale(p.predict(images)))
    

if __name__ == '__main__':
    main()