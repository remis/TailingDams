# TailingDamDetection

## To do
1. Check false positive on the data the neural net used, especially for areas where for similar subareas there are no detections. E.g. 16 12' 48.86''S 47 31' 31.82''W

## Data changes
* 13 February 2019 - initial data. 
- Three categories: dams (539 files), no dams (300 files), look-a-likes (51 files). These numbers based on saved numpy arrays. Currently in the folder the number of dams is different - 531. 
- Area approximately 50km^2, or 236x236 pixels (minimum values)

* 19 March 2019 - updated dam images, smaller areas. After the first experiments it has been noticed that 50km^2 areas are too big and dams occupy just a small fraction of images, and images may contain several dams. It has been decided that the area should be dramaticallty decreased such that dams would dominate images. The area of images was reduced, also small dams were discarded from consideration.
- New images for dams (855 files)
- Image patches with a diagonal of 4km, or 95x95 pixels (minimum values)
- For look-a-likes the central 95x95 area was cropped from the old images
- For no dams two random patches of size 95x95 were cropped from the old images (2 to keep data balanced since the number of dams had increased)

* 9 April 2019 - updated dam and no dam images. New data was created to addressed mistakes made by the neural network trained on the previous version of the data. Main points of misclassification were identified in urban and mountain areas.
- New images of dams (800 files) and no dams (402 files)
- Area approximately 4km^2, or 134x134 pixels (minimum values)
- 52 dams were removed located in the urban areas
- Smaller areas for old no dams "bases" plus 100 new no dams areas in urban and mountain areas and areas of false positives by the neural network trained on the previous version of data
