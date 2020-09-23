# import the necessary packages
from PIL import Image
import pytesseract
import argparse
import cv2
import os
import subprocess
from googletrans import Translator
from PIL.ExifTags import TAGS

def getImageMetaData(imagePath):
	infoDict = dict()
	exifToolPath = "exiftool"
	process = subprocess.Popen([exifToolPath,imagePath],stdout=subprocess.PIPE, stderr=subprocess.STDOUT,universal_newlines=True) 
	''' get the tags in dict '''
	for tag in process.stdout:
		line = tag.strip().split(':')
		infoDict[line[0].strip()] = line[-1].strip()

	for k,v in infoDict.items():
		print(k,':', v)

def returnImage(imagePath):
	if os.path.isfile(imagePath):
		image = cv2.imread(imagePath)
	else:
		raise FileNotFoundError

	return image

def preprocessImage(image, preprocess = "thresh"):
	# check to see if we should apply thresholding to preprocess the
	# image	
	if preprocess == "thresh":
		gray = cv2.threshold(image, 0, 255,
			cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	# make a check to see if median blurring should be done to remove
	# noise
	elif preprocess == "blur":
		gray = cv2.medianBlur(image, 3)

	return gray

def getImageText(imagePath, output, options, useOptions = True):
	if useOptions:
		text = pytesseract.image_to_string(Image.open(imagePath), config=options)
	else:
		text = pytesseract.image_to_string(Image.open(imagePath))
	if output == "stdout":
		print(text)
	else:
		fileObj = open(output,mode = "w")
		fileObj.write(text)
	
	return text

def showScaledImage(image, height=800, name = "Image") :
	try:
		imHeight, imWidth, _ = image.shape
	except:
		imHeight, imWidth = image.shape
	
	scaleRatio = imHeight/height
	width = int(imWidth / scaleRatio)
	
	imageResize = cv2.resize(image, (width,height))
	gray
	cv2.imshow(name, imageResize)

	return width, height

def showRoundingBoxes(imagePath, width, height, options, useOptions = True):
	img = cv2.imread(imagePath)
	if useOptions:
		data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=options)
	else:
		data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
	n_boxes = len(data['text'])
	for i in range(n_boxes):
		if int(data['conf'][i]) > 60:
			(x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
			img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

	imageResize = cv2.resize(img, (width,height))
	cv2.imshow('Boxes', imageResize)
	cv2.waitKey(0)

def translateContent(inputType, inputString = "", filename = "", writeToFile = True, writeFile = ""):
	translator = Translator()
	if inputType == "stdout":
		detectedLang = translator.detect(inputString)
		translated = translator.translate(inputString)
	else:
		with open(filename, "r") as fileObj:
			data = fileObj.readlines()
		data = ''.join(data)
		detectedLang = translator.detect(text = data)
		translated = translator.translate(data)


	print("Your file has content written in {}, we can say that with a confidence of {}%".format(detectedLang.lang, detectedLang.confidence * 100))

	if len(writeFile):
		if writeToFile:	
			with open(writeFile,'w') as fileObj:
					fileObj.write(translated.text)
		else:
			print(translated.text)
	else:
		print("File name not provided, exiting")
		os._exit(0)

	return None


if __name__ == "__main__":

	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True,
		help="path to input image to be OCR'd")
	ap.add_argument("-re", "--preprocess", type=str, default="thresh",
		help="type of preprocessing to be done")
	ap.add_argument("-f", "--file", type=str, default="stdout",
		help="output file, stdout by default")
	ap.add_argument("-m", "--meta", type=int, default=0,
		help="metadata, 0 for no, anything else for yes")
	ap.add_argument("-l", "--lang", type=str, default="eng",
		help="language that Tesseract will use when OCR'ing")
	ap.add_argument("-t", "--to", type=str, default="en",
		help="language that we'll be translating to")
	ap.add_argument("-p", "--psm", type=int, default=13,
		help="Tesseract PSM mode")
	ap.add_argument("-u", "--use", type=bool, default=False,
		help="Use special config options")	
	ap.add_argument("-w", "--translatedfile", type=str, default="",
		help="Write the translated content into this file")	
	args = vars(ap.parse_args())

	imagePath = args["image"]
	options = "-l {} --psm {}".format(args["lang"], args["psm"])
	useOptions = args["use"]

	image = returnImage(imagePath)

	height, width, _ = image.shape
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	preprocess = args["preprocess"]

	preprocessedImage = preprocessImage(gray,preprocess)
	# write the grayscale image to disk as a temporary file so we can
	# apply OCR to it

	filename = "{}.png".format("outfile")
	cv2.imwrite(filename, preprocessedImage)

	output = args["file"]
	text = getImageText(imagePath, output, options, useOptions)

	width, height = showScaledImage(image, name = "Original", height=400)
	showScaledImage(preprocessedImage, name = "Preprocessed", height=400)

	showRoundingBoxes(imagePath, width, height, options, useOptions)

	writeFile = args["translatedfile"]
	translateContent(output, filename="text", writeFile=writeFile)

	if args["meta"] == 1:
		getImageMetaData(imagePath)
	