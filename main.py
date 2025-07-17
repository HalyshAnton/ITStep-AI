import onnxruntime
import torch
from torchvision import transforms
from PIL import Image

classes = ['Avulsion fracture',
 'Comminuted fracture',
 'Fracture Dislocation',
 'Greenstick fracture',
 'Hairline Fracture',
 'Impacted fracture',
 'Longitudinal fracture',
 'Oblique fracture',
 'Pathological fracture',
 'Spiral Fracture']

path = 'model.onnx'

session = onnxruntime.InferenceSession(path)

# завантаження зображення
image = Image.open('data/lesson_many/fracture_dislocation.jpg')
# image.show()

# обробка зображення

# Визначити конвеєр перетворень
transformer = transforms.Compose([
    transforms.Resize((64, 64)),   # зміна розміру зображення на 64х64 пікселя
    transforms.ToTensor(),  # перевести в тензор
])

image = transformer(image)

# добавити 1 до розміру тензора
image = image.unsqueeze(0)  # 1 має бути під індексом 0

# перевести у масив numpy
image = image.numpy()

# print(image.shape)
# print(image)


results = session.run(
    ['result'],   # назви результатів які треба отримати, або None щоб отримати все
    {'image': image}
)

# отримати результат для першого зображення
result = results[0]

# перевести результат в ймовірності
result = torch.tensor(result)  # переводимо назад у тензор
result = torch.nn.functional.softmax(result, dim=1)
result = result.numpy()

print(result.shape)
print(result)

# отримати максимальну ймовірність та її індекс
result = result[0]  # дістаємо дані для першого(єдиного зображення)

max_proba = result.max()
max_idx = result.argmax()  # індекс максимального елемента
image_class = classes[max_idx]  # назва класу для зображення

print(f'Максимальна ймовірність: {max_proba}')
print(f'Індекс: {max_idx}')
print(f'Назва перелому: {image_class}')