import torch
import json
import os

def save_model(model, optimizer, path: str, epoch: int):

    # Salvar o estado do modelo e do otimizador junto com a época
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }
    torch.save(state, path)
    print(f"Modelo e estado salvos em: {path}")

def load_model(model, optimizer, path: str):

    if not os.path.exists(path):
        raise FileNotFoundError(f"O arquivo {path} não foi encontrado.")
    
    # Carregar o estado salvo
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    epoch = checkpoint["epoch"]

    print(f"Modelo carregado de: {path}")
    print(f"Época carregada: {epoch}")
    
    return model, optimizer, epoch
