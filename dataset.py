import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split


class Xray(Dataset):
	def __init__(self, path, transform=None, is_train=True):
		class_list = os.listdir(path)
		class_path = [os.path.join(path, class_name) for class_name in class_list]

		self.normal_list = [class_path[0] + '/' + file for file in os.listdir(class_path[0])]
		self.pneumonia_list = [class_path[1] + '/' + file for file in os.listdir(class_path[1])]
		self.covid19_list = [class_path[2] + '/' + file for file in os.listdir(class_path[2])]


		normal_train, normal_test = train_test_split(self.normal_list)
		pneumonia_train, pneumonia_test = train_test_split(self.pneumonia_list)
		covid19_train, covid19_test = train_test_split(self.covid19_list)

		self.path = path

		if is_train:
			self.data_path = normal_train + pneumonia_train + covid19_train
		else:
			self.data_path = normal_test + pneumonia_test + covid19_test

		if transform is None:
			self.transform = transforms.Compose([transforms.ToTensor()])
		else:
			self.transform = transform
		
		self.label_map ={"COVID19":0, "NORMAL":1, "PNEUMONIA":2}

	def __len__(self):
		return len(self.data_path)

	def __getitem__(self, index):
		image = Image.open(self.data_path[index]).convert("RGB")
		image = self.transform(image)
		label_name = self.data_path[index].split("/")[1]
		label = self.label_map[label_name]
		return image,label


if __name__ == "__main__":
	transform = transforms.Compose([
		transforms.Resize((256, 256)),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

	dataset = Xray("data", transform=transform, is_train=True)

	for i in range(5):
		image, label = dataset[i]
		print(image.shape, label)
