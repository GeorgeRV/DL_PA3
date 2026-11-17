from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.dataset import ImagePathIndex, load_class_mapping, TrapDataset
from src.model import build_model
from torchvision import transforms
from src.dataset import SquarePad

def main():
    ap=argparse.ArgumentParser("Diagnóstico de pseudo-rótulos no test_public (somente diagnóstico)")
    ap.add_argument("--data-dir",type=Path,default=Path("data/pa3_dados"))
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--thresholds",type=float,nargs="+",default=[0.50,0.70,0.80,0.90,0.95,0.99])
    ap.add_argument("--batch-size",type=int,default=32)
    ap.add_argument("--num-workers",type=int,default=2)
    ap.add_argument("--device",default="auto")
    a=ap.parse_args()
    device=torch.device("cuda" if a.device=="auto" and torch.cuda.is_available() else
                        ("cpu" if a.device=="auto" else a.device))
    ld=a.data_dir/"labels"; names,aliases=load_class_mapping(a.data_dir/"classes.csv",[ld/"test_public.csv"])
    idx=ImagePathIndex(a.data_dir/"images")
    tf=transforms.Compose([SquarePad(),transforms.Resize((224,224)),transforms.ToTensor(),
                           transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
    ds=TrapDataset(ld/"test_public.csv",idx,aliases,"test_public",tf)
    loader=DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=a.num_workers)
    ck=torch.load(a.checkpoint,map_location="cpu")
    model=build_model("resnet18",len(names),pretrained=False); model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    probs_all=[]; y_all=[]
    with torch.no_grad():
        for x,y,_ in loader:
            probs_all.append(torch.softmax(model(x.to(device)),1).cpu().numpy()); y_all.extend(y.numpy().tolist())
    probs=np.concatenate(probs_all); y=np.asarray(y_all); pred=probs.argmax(1); conf=probs.max(1)
    rows=[]
    for t in a.thresholds:
        selected=conf>=t
        for c,name in enumerate(names):
            assigned=selected & (pred==c)
            true_class=(y==c)
            correct=assigned & true_class
            precision=float(correct.sum()/assigned.sum()) if assigned.sum() else float("nan")
            coverage=float(correct.sum()/true_class.sum()) if true_class.sum() else float("nan")
            rows.append({"threshold":t,"class_idx":c,"class_name":name,
                         "pseudo_labels_assigned":int(assigned.sum()),
                         "true_class_examples":int(true_class.sum()),
                         "correct_pseudo_labels":int(correct.sum()),
                         "precision":precision,"coverage":coverage,
                         "class_test_support":int(true_class.sum())})
    a.output_dir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(a.output_dir/"pseudo_label_precision_coverage_by_class.csv",index=False)
    dist=[]
    for t in a.thresholds:
        sel=conf>=t
        counts=np.bincount(pred[sel],minlength=len(names))
        for c,name in enumerate(names):
            dist.append({"threshold":t,"class_idx":c,"class_name":name,
                         "accepted_pseudo_labels":int(counts[c]),
                         "fraction_of_accepted":float(counts[c]/max(sel.sum(),1)),
                         "class_prior_in_labeled_budget":1/len(names)})
    pd.DataFrame(dist).to_csv(a.output_dir/"pseudo_label_distribution.csv",index=False)
    print("Diagnóstico concluído. ATENÇÃO: test_public foi usado apenas para diagnóstico, não para seleção.")
    print(a.output_dir/"pseudo_label_precision_coverage_by_class.csv")

if __name__=="__main__": main()
