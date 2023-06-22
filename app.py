from dash import Dash, Input, Output, html, dcc
from PIL import Image, UnidentifiedImageError
import numpy as np
import plotly.express as px
import pandas as pd
from torchvision import transforms
import torch
from CustomNN import ImageNN

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

app.layout = html.Div(
    [
        html.H1("Recognize digits based on handwritten images"),

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
def open_image(filename):
    try:
        image = np.array(Image.open(filename))
    except UnidentifiedImageError:
        return html.Div(
            [
                "Please load a valid image file!!"
            ]
        )

    return image


device = ('cuda' if torch.cuda.is_available() else 'cpu')


# predicted image digit

# def predicted_digit(image, model=model):
#     image = transform(image).to(device)
#
#     output = model(image)
#     _, label = torch.max(output.data, 1)
#
#     return label.item()


def predicted_digit(image):
    # Transform the image
    image = transform(image).unsqueeze(0).to(device)

    # Perform the prediction
    with torch.no_grad():
        model.eval()
        output = model(image)

    # Get the predicted digit
    _, predicted = torch.max(output.data, 1)
    digit = predicted.item()

    return digit


# predicted image digit


@app.callback(
    Output('show-image', 'children'),
    Input('upload-image', 'filename')
)
def show_image(filename):
    if filename is not None:
        image = open_image(filename)
        if isinstance(image, np.ndarray):

            figure = px.imshow(image, color_continuous_scale='gray')
            figure.update_layout(coloraxis_showscale=False)
            figure.update_xaxes(showticklabels=False)
            figure.update_yaxes(showticklabels=False)

            return html.Div(
                [
                    dcc.Graph(figure=figure),

                    html.H1(f"Predicted digit: {predicted_digit(Image.open(filename))}")

                ]
            )
        else:
            return image


if __name__ == "__main__":
    app.run_server(debug=True)
