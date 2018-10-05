import argparse
import cv2
import pytesseract
import numpy as np
import json
import math

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
				help="path to input image to be OCR'd")
args = vars(ap.parse_args())

# 2D distance between 2 points (x1, y1) (x2, y2)
def distance(x1, y1, x2, y2):
    return math.sqrt(math.pow((x2-x1), 2) + math.pow((y2-y1), 2))

im = cv2.imread(args['image'])

img2gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 20, 255, cv2.THRESH_BINARY)

im2, contours, hierarchy = cv2.findContours(
	mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Rectangle for each question:
rects = []
for c in contours:
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	x, y, w, h = cv2.boundingRect(approx)

	if(h >= 30 and h <= 100):
		# if height is enough
		# create rectangle for bounding
		rect = (x, y, w, h)
		rects.append(rect)

length = len(rects)

# allow margin of error in circle detection
padding = 1

choices = ['a','b','c','d']

# output
answers = []

for j in range(0, length, 2):
	f = rects[j + 1]
	roi = im[(f[1] - padding):(f[1] + f[3] + padding),
			  (f[0] - padding):(f[0] + f[2] + padding)]
	roigrey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(roigrey, 127, 255, 0)
	roi2, rcontours, hierarchy = cv2.findContours(
		thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	circles = cv2.HoughCircles(roigrey, cv2.HOUGH_GRADIENT, 1, 20,
						param1=50, param2=30, minRadius=0, maxRadius=0)

	# answers
	values = []
	if circles is not None:
		circles = np.uint16(np.around(circles))

		for inde,i in enumerate(circles[0, :]):
			# draw the outer circle
			circlerect = roi[(i[1]-i[2]):(i[1]+i[2]), (i[0]-i[2]):(i[0]+i[2])]
			width, height, depth = circlerect.shape
			radius = i[2]
			mask = np.zeros((width, height), np.uint8)
			cv2.circle(mask,(radius, radius), radius, (255, 255, 255), -1)

			circle =  cv2.bitwise_and(circlerect, circlerect, mask=mask)
			sum = 0;
			for x in range(0,width-1):
				for y in range(0,height-1):
					if(utilities.distance(x, y, radius, radius) <= radius):
						color = np.sum(circle[x,y])
						if(color < 450):
							sum+=1
			sum = sum/(width*height)
			if(sum >= 0.5):
				obj = {"color": "black", "pos": i[0]}
				values.append(obj)
			else:
				obj = {"color": "white", "pos": i[0]}
				values.append(obj)

	#sort values
	values = sorted(values, key=lambda k: k['pos'])
	ans = "no answer"
	for idx, val in enumerate(values):
		if(val["color"] == "black"):
			ans=choices[idx]

	# question numbers
	numbers = []

	for rc in rcontours:
		# get question number
		roiperi = cv2.arcLength(rc, True)
		roiapprox = cv2.approxPolyDP(rc, 0.02 * roiperi, True)
		roix, roiy, roiw, roih = cv2.boundingRect(roiapprox)

		if (roih >= 10 and roih <= 20):
			imgnum = roi[roiy:roiy+roih, roix:roix+roiw]
			strnum = pytesseract.image_to_string(imgnum, config='-psm 6')
			numbers.append(strnum)

	questionNumber = 0
	try:
		questionNumber = int(''.join(list(reversed(numbers))))
	except:
		questionNumber = -1

	answer = {"num": questionNumber, "ans": ans}
	answers.append(answer)

answers = sorted(answers, key=lambda k: k['num'])

with open('output.json', 'w') as fout:
    json.dump(answers, fout)
