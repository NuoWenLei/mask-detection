from model import Mask_Predictor
from mtcnn import MTCNN
from PIL import Image, ImageDraw
import tkinter as tk, cv2, numpy as np, keras

image_x = 299
image_y = 299
window = tk.Tk()

def preprocess_image(image, net_h, net_w):
    new_h, new_w, _ = image.shape

    # determine the new size of the image
    if (float(net_w)/new_w) < (float(net_h)/new_h):
        new_h = (new_h * net_w)//new_w
        new_w = net_w
    else:
        new_w = (new_w * net_h)//new_h
        new_h = net_h
    
    # Try resizing image, sometimes there are images that trigger errors related to cv2's source code, which we can filter out
    try:
      resized = cv2.resize(image/255., (int(new_w), int(new_h)))
      
      # Create gray background with right size
      new_image = np.ones((net_h, net_w, 3)) * 0.5

      # Overlay resized image on gray background, essentially padding the image to fit the net size
      new_image[int((net_h-new_h)//2):int((net_h+new_h)//2), int((net_w-new_w)//2):int((net_w+new_w)//2), :] = resized
      new_image = np.expand_dims(new_image, 0)
      return new_image
    except Exception as e:
      print(e)
      return

def preprocess_serving(f):

	# Instantiate image list
	X = []


	# Instantiate MTCNN detector
	# Learn more at: https://github.com/ipazc/mtcnn
	detector = MTCNN()

	# # Loop through each provided file path
	# for f in files:

	# Read image as 2D numpy array
	im = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)

	# Detect faces in image
	faces = detector.detect_faces(im)
	if len(faces) == 0:
		print(f"File {f} Contains No Faces")
		return

	# Loop through each face
	for face in faces:

		# Crop image to contain one face
		far_x = face['box'][2] + face['box'][0]
		far_y = face['box'][3] + face['box'][1]
		im_arr = im[face['box'][1]:far_y, face['box'][0]:far_x, :3]
		if 0 in im_arr.shape:
			print("Invalid Face, Please Try Another Image")
			return

		# Pad image to fit correct size
		process_res = preprocess_image(im_arr, image_y, image_x)
		if process_res is None:
			print("Invalid Face, Please Try Another Image")
			return
		X.append(process_res)
	return np.array(X), faces

def draw_faces(f, boxes, preds, confidence=1):
	im = Image.open(f)
	imDraw = ImageDraw.Draw(im)
	for i, box in enumerate(boxes):
		color = "green" if preds[i][0] else "red"
		imDraw.rectangle([(box[0], box[1]), (box[0]+box[2], box[1]+box[3])], outline=color)
		if confidence:
			imDraw.rectangle([(box[0], box[1]), (box[0]+120, box[1]-20)], fill=color)
			imDraw.text([box[0]+2, box[1]-15], f"Confidence: {preds[i][1]:.2f}", fill="white")

	return im

counter = 0

def predict():
	global counter
	filename = entry.get()
	data, faces = preprocess_serving(filename)

	# Reshape data for prediction convenience
	data = data.reshape(data.shape[0], image_y, image_x, 3)

	# Predict data with model
	pred_prob = model.predict(data)

	# Do basic response and create binary prediction list
	preds = []
	for prob in pred_prob:
		pred = np.argmax(prob)
		preds.append([pred, prob[pred]])
		if pred:
			print("You are wearing a mask")
		else:
			print("You are not wearing a mask")

	# Draw bounding boxes of faces on image
	im_res = draw_faces(filename, [i['box'] for i in faces], preds, confidence=0)
	im_res.save(f'../pictures/result_image{counter}.png')
	counter += 1
	im_res.show()


model = keras.models.load_model('../mask_predictor.h5')
label = tk.Label(text="image path")
entry = tk.Entry()
button = tk.Button(text="find faces", command=predict)
label.pack()
entry.pack()
button.pack()
window.mainloop()

