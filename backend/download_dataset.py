import os
import requests
from PIL import Image
from io import BytesIO

# Create directories if they don't exist
os.makedirs('dataset/cats', exist_ok=True)
os.makedirs('dataset/dogs', exist_ok=True)

# Sample image URLs (5 cats and 5 dogs from reliable sources)
cat_urls = [
    'https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/4-Classification/1-Introduction/images/cat.jpg',
    'https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/4-Classification/2-Classifiers-1/images/persian.jpg',
    'https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/4-Classification/2-Classifiers-1/images/siamese.jpg',
    'https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/4-Classification/2-Classifiers-1/images/tabby.jpg',
    'https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/4-Classification/2-Classifiers-1/images/calico.jpg'
]

dog_urls = [
    'https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/4-Classification/1-Introduction/images/dog.jpg',
    'https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/4-Classification/2-Classifiers-1/images/husky.jpg',
    'https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/4-Classification/2-Classifiers-1/images/retriever.jpg',
    'https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/4-Classification/2-Classifiers-1/images/german-shepherd.jpg',
    'https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/4-Classification/2-Classifiers-1/images/pug.jpg'
]

def download_and_save_image(url, filename):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')  # Convert to RGB format
        img = img.resize((224, 224))  # Resize to match our model's input size
        img.save(filename)
        print(f"Downloaded: {filename}")
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")

# Download cat images
for i, url in enumerate(cat_urls):
    download_and_save_image(url, f'dataset/cats/cat_{i}.jpg')

# Download dog images
for i, url in enumerate(dog_urls):
    download_and_save_image(url, f'dataset/dogs/dog_{i}.jpg')

print("Dataset download completed!") 