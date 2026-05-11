from src.storage import load_model, load_test_dataloader

if __name__ == '__main__':
    model = load_model('results/models/example.pth')
    dataloader_test = load_test_dataloader('results/models/example.pth')
    print(model)
    print(dataloader_test)