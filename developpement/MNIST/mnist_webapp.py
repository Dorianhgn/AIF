import gradio as gr
from PIL import Image
import requests
import io


def recognize_digit(image):
    # Extract the composite image from the sketchpad dictionary
    if isinstance(image, dict) and 'composite' in image:
        image = image['composite']
    
    # Convert to PIL Image
    if hasattr(image, 'astype'):
        # If it's a numpy array
        image = Image.fromarray(image.astype('uint8'))
    else:
        # If it's already a PIL Image
        image = image
    
        # Convert to grayscale to ensure single channel
    try:
        if image.mode != 'L':
            image = image.convert('L')
    except Exception as e: # pass any exception
        print(f"Error converting image to grayscale: {e}")
        return -1

    # revert image colors
    image = Image.eval(image, lambda x: 255 - x)

    img_binary = io.BytesIO()
    image.save(img_binary, format='PNG')
    img_binary = img_binary.getvalue()

    # response = requests.post("http://mnist-api:5075/predict", data=img_binary, headers={"Content-Type": "application/octet-stream"}) # if using `docker network create mnist-network`
    response = requests.post("http://api:5075/predict", data=img_binary, headers={"Content-Type": "application/octet-stream"}) # Use localhost if running docker-compose
    if response.ok:
        prediction = response.json().get("prediction", -1)
    else:
        print(f"Request failed with status {response.status_code} - {response.reason}")
        print("Response body:", response.text)
        prediction = -1

    return prediction

if __name__=='__main__':

    gr.Interface(fn=recognize_digit, 
                inputs="sketchpad", 
                outputs='label',
                live=True,
                description="Draw a number on the sketchpad to see the model's prediction.",
                ).launch(debug=True, share=True);