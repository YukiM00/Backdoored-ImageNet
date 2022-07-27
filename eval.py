import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def eval(x,x_b,y,model,targeted=-1):
    print("CLEAN")
    trues = np.argmax(y,axis=1)
    preds_clean = np.argmax(model.predict(x), axis=1)
    matrix_clean = confusion_matrix(trues,preds_clean)
    print("Error rate:", 1- accuracy_score(trues,preds_clean))
    print(matrix_clean)

    print("TRIGGER")
    preds_poison = np.argmax(model.predict(x_b), axis=1)
    matrix_poison = confusion_matrix(trues,preds_poison)
    print("Error rate:", 1- accuracy_score(trues,preds_poison))
    print(matrix_poison)
    
    asr_list =[]
    
    for i in range(y.shape[1]):
        asr = 0
        for j in range(y.shape[1]):
            asr = asr + matrix_poison[j,i]
        asr = asr/x_b.shape[0]
        asr_list.append(asr)

    if targeted == -1:
        print("Rf list",asr_list)
    else:
        print("Rf class",targeted,":",asr_list[targeted])
    
    return matrix_clean,matrix_poison
