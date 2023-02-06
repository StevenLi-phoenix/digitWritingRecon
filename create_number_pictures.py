import os
from PIL import Image, ImageDraw, ImageFont

fontsize = 32

for i in range(10):
    im = Image.new("RGB", (64, 64), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    font = ImageFont.truetype("static/fonts/msyh.ttc", fontsize)
    dr.text((22, 12), str(i),font=font,fill="#000000")
    im.save(os.path.join("digits","%s.jpg"%i))