# Detailed Evaluation Function

Replace the default `evaluate()` in `Image_Classification.py` with this version to get **per-class accuracy** and a **confusion matrix**.

## Usage
Copy and replace the `evaluate()` function in `Image_Classification.py`:

```python
def evaluate(test_data):
    dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    model = ImageModel()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model.load_state_dict(torch.load(os.path.join(base_dir, 'model', 'model.pth'), weights_only=True))

    class_names = ['Down', 'Fist', 'Left', 'Right', 'Up']
    correct_per_class = [0] * 5
    total_per_class   = [0] * 5
    confusion = [[0]*5 for _ in range(5)]

    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            y_pred = torch.argmax(model(x), dim=-1)
            for true, pred in zip(y, y_pred):
                confusion[true][pred] += 1
                total_per_class[true] += 1
                if true == pred:
                    correct_per_class[true] += 1

    print(f'\n{"Class":<8} {"Correct":>8} {"Total":>8} {"Accuracy":>10}')
    print("-" * 38)
    for i, name in enumerate(class_names):
        acc = correct_per_class[i] / total_per_class[i]
        print(f'{name:<8} {correct_per_class[i]:>8} {total_per_class[i]:>8} {acc:>10.2%}')

    total_correct = sum(correct_per_class)
    total_sample  = sum(total_per_class)
    print(f'\nOverall accuracy: {total_correct/total_sample:.4f}')

    print(f'\nConfusion Matrix (row=actual, col=predicted):')
    print(f'{"":>8}', end='')
    for name in class_names:
        print(f'{name:>8}', end='')
    print()
    for i, name in enumerate(class_names):
        print(f'{name:>8}', end='')
        for j in range(5):
            print(f'{confusion[i][j]:>8}', end='')
        print()
```

## Output Example
```
Class     Correct    Total   Accuracy
--------------------------------------
Down          198      200     99.00%
Fist          196      200     98.00%
Left          197      200     98.50%
Right         195      200     97.50%
Up            199      200     99.50%

Overall accuracy: 0.9700

Confusion Matrix (row=actual, col=predicted):
            Down    Fist    Left   Right      Up
    Down     198       2       0       0       0
    Fist       0     196       3       0       1
    Left       0       1     197       2       0
   Right       0       0       2     195       3
      Up       0       1       0       0     199
```