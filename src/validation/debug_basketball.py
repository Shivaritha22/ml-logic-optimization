from PIL import Image
import matplotlib.pyplot as plt

img = Image.open(r"C:\Users\shiva\Shiv\enee759\ml-logic-optimization\data\widerface\WIDER_val\images\35--Basketball\35_Basketball_basketballgame_ball_35_709.jpg")
print(img.size)
plt.imshow(img)
plt.savefig("basketball_check.png")