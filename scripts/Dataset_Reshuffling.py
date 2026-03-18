import os, shutil, random

base = r"C:\Users\aayn0\OneDrive\Desktop\Project\AI-Based Hand Gesture Controlled Robotic Car\Dataset"
classes = ['Down', 'Fist', 'Left', 'Right', 'Up']

for cls in classes:
    train_dir = os.path.join(base, 'Train', cls)
    test_dir  = os.path.join(base, 'Test', cls)
    
    # 把Train和Test的图片全部合并
    all_imgs = []
    for img in os.listdir(train_dir):
        shutil.move(os.path.join(train_dir, img), os.path.join(train_dir, f'tr_{img}'))
        all_imgs.append(os.path.join(train_dir, f'tr_{img}'))
    for img in os.listdir(test_dir):
        shutil.move(os.path.join(test_dir, img), os.path.join(test_dir, f'te_{img}'))
        all_imgs.append(os.path.join(test_dir, f'te_{img}'))
    
    # 随机打乱
    random.shuffle(all_imgs)
    split = int(len(all_imgs) * 0.8)
    
    # 重新分配
    for i, src in enumerate(all_imgs[:split]):
        shutil.move(src, os.path.join(train_dir, f'{i+1}.jpg'))
    for i, src in enumerate(all_imgs[split:]):
        shutil.move(src, os.path.join(test_dir, f'{i+1}.jpg'))
    
    print(f'{cls}: {split} train, {len(all_imgs)-split} test')

print('Done!')