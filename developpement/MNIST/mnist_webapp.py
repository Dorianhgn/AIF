import gradio as gr
from PIL import Image
import requests
import io


def recognize_digit(image):
    # Convert to PIL Image necessary if using the API method
    image = Image.fromarray(image.astype('uint8'))
    img_binary = io.BytesIO()
    image.save(img_binary, format='PNG')
    img_binary = img_binary.getvalue()

    response = requests.post("http://127.0.0.1:5000/predict", data=img_binary, headers={"Content-Type": "application/octet-stream"})
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