import torch
import os

def save_parameter(model, path):
    if os.path.exists(path):
        os.remove(path)
    torch.save({'model': model.state_dict()}, path)



def load_paramater(new_model, path):
    p = torch.load(path)
    new_model.load_state_dict(p['model'], strict=False)
    return new_model

'''


a = Model()
path = '../../zhenghuo1.model'

if os.path.exists(path):
    os.remove('../../zhenghuo1.model')

torch.save({'model': a.state_dict()}, '../../zhenghuo1.model')

print(a.state_dict())

d= newModel()
p= torch.load('../../zhenghuo1.model')
d.load_state_dict(p['model'],strict=False)
print(d.state_dict())

#print(nn.Linear(2,2).parameters())
#print(a.state_dict())
'''
