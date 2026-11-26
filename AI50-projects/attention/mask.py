import os
import sys

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, TFBertForMaskedLM


MODEL = "bert-base-uncased"


K = 3


FONT_PATH = os.path.join(os.path.dirname(__file__), "assets", "fonts", "OpenSans-Regular.ttf")
FONT = ImageFont.truetype(FONT_PATH, 28)
GRID_SIZE = 40
PIXELS_PER_WORD = 200


def main():
    text = input("Text: ")


    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    inputs = tokenizer(text, return_tensors="tf")
    mask_token_index = get_mask_token_index(tokenizer.mask_token_id, inputs)
    if mask_token_index is None:
        sys.exit(f"Input must include mask token {tokenizer.mask_token}.")


    model = TFBertForMaskedLM.from_pretrained(MODEL)
    result = model(**inputs, output_attentions=True)


    mask_token_logits = result.logits[0, mask_token_index]
    top_tokens = tf.math.top_k(mask_token_logits, K).indices.numpy()
    for token in top_tokens:
        print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))


    visualize_attentions(inputs.tokens(), result.attentions)


def get_mask_token_index(mask_token_id, inputs):

    input_ids = inputs["input_ids"][0].numpy()
    for index, token_id in enumerate(input_ids):
        if int(token_id) == mask_token_id:
            return index
    return None



def get_color_for_attention_score(attention_score):

    shade = int(float(attention_score) * 255)
    return (shade, shade, shade)



def visualize_attentions(tokens, attentions):

    for layer_index, layer_attention in enumerate(attentions):
        for head_index, attention_weights in enumerate(layer_attention[0]):
            generate_diagram(
                layer_index + 1,
                head_index + 1,
                tokens,
                attention_weights
            )


def generate_diagram(layer_number, head_number, tokens, attention_weights):


    image_size = GRID_SIZE * len(tokens) + PIXELS_PER_WORD
    img = Image.new("RGBA", (image_size, image_size), "black")
    draw = ImageDraw.Draw(img)


    for i, token in enumerate(tokens):

        token_image = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
        token_draw = ImageDraw.Draw(token_image)
        token_draw.text(
            (image_size - PIXELS_PER_WORD, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )
        token_image = token_image.rotate(90)
        img.paste(token_image, mask=token_image)


        _, _, width, _ = draw.textbbox((0, 0), token, font=FONT)
        draw.text(
            (PIXELS_PER_WORD - width, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )


    for i in range(len(tokens)):
        y = PIXELS_PER_WORD + i * GRID_SIZE
        for j in range(len(tokens)):
            x = PIXELS_PER_WORD + j * GRID_SIZE
            color = get_color_for_attention_score(attention_weights[i][j])
            draw.rectangle((x, y, x + GRID_SIZE, y + GRID_SIZE), fill=color)

    
    img.save(f"Attention_Layer{layer_number}_Head{head_number}.png")


if __name__ == "__main__":
    main()
