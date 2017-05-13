from PIL import Image


#Cropping face out of background
def crop_face(image, offset_pct=(0.2,0.2), crop = (150,150)):
  width, height = image.size

  pixelsToCropTopBottom = int(height*offset_pct[1])
  pixelsToCropLeftRight = int(width*offset_pct[0])

  leftPixel = pixelsToCropLeftRight
  upperPixel = pixelsToCropTopBottom
  rightPixel = width - pixelsToCropLeftRight
  lowerPixel =  height - pixelsToCropTopBottom
  image = image.crop((leftPixel, upperPixel, rightPixel, lowerPixel)) #4-tuple defining the left, upper, right, and lower pixel

  resizeDim = (crop[0], crop[1])
  image = image.resize(resizeDim, Image.ANTIALIAS)

  return image


if __name__ == "__main__":
  image = Image.open("/home/deeplearning/PR-Gender-Recognition/result_faces/Donald Trump/ATACAN A DONALD TRUMP , ULTIMO MINUTO, HOY 8 DE MAYO 2017/face20.jpg")
  image.save("test.jpg")
  crop_face(image,  offset_pct=(0.1,0.1), crop=(200,200)).save("result.jpg")