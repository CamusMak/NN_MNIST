from dash import Dash, Input, Output, html, dcc
from PIL import Image, UnidentifiedImageError
import numpy as np
import plotly.express as px
import pandas as pd
from torchvision import transforms
import torch
import io
import base64
from CustomNN import ImageNN, predicted_digit, open_image

# load the model
model = ImageNN()
model.state_dict = torch.load("model_1.pth")

# image transformation
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

app = Dash(__name__)

server = app.server
app.layout = html.Div(
    [
        html.H1("Handwritten digit recognition"),

        dcc.Upload(
            id='upload-image',
            children=[
                html.A('Upload an image')
            ],
            style={
                'width': '20%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            }

        ),

        html.Div(
            id='show-image'
        )

    ]
)


# open image with Image module from PIL

@app.callback(
    Output('show-image', 'children'),
    Input('upload-image', 'contents'),
    prevent_initial_call=True
)
def process_uploaded_image(contents):
    try:
        image = Image.open(io.BytesIO(base64.b64decode(contents.split(',')[1])))
    except UnidentifiedImageError:
        return html.Div(
            [
                "Please load a valid image file!!"
            ]
        )
    image_tensor = transform(image).unsqueeze(0)

    predicted_class = None
    # Predict the digit
    with torch.no_grad():
        output = model(image_tensor)
        predicted_class = torch.argmax(output).item()
        print(predicted_class)

    return html.Div(
        [
            # html.H2(f"Predicted digit: {predicted_digit(model=model,image=image)}"),
            html.H2(f"Predicted digit: {predicted_class}"),
            html.Img(src=contents, style={'width': '400px', 'height': '400px'})
        ]
    )




if __name__ == "__main__":
    app.run_server(debug=True)
