from __future__ import annotations
import argparse, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from src.dataset import load_datasets
from src.fixmatch_data import load_fixmatch_datasets
from src.model import build_model, count_parameters
from src.metrics import save_training_curves

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def evaluate(model, loader, device, nclasses):
    model.eval(); ys=[]; ps=[]; loss_sum=0.; n=0
    criterion=nn.CrossEntropyLoss()
    criterion_per_example=nn.CrossEntropyLoss(reduction="none")
    with torch.no_grad():
        for x,y,_ in loader:
            x=x.to(device); y=y.to(device); logits=model(x)
            loss_sum += criterion(logits,y).item()*len(y); n+=len(y)
            ys.extend(y.cpu().tolist()); ps.extend(logits.argmax(1).cpu().tolist())
    return {"loss":loss_sum/max(n,1),
            "f1_macro":float(f1_score(ys,ps,labels=list(range(nclasses)),average="macro",zero_division=0)),
            "accuracy":float(accuracy_score(ys,ps))}

def main():
    ap=argparse.ArgumentParser("FixMatch para PA3")
    ap.add_argument("--data-dir",type=Path,default=Path("data/pa3_dados"))
    ap.add_argument("--output-dir",type=Path,default=Path("outputs/parte1/fixmatch"))
    ap.add_argument("--method",choices=["standard","aligned"],default="standard")
    ap.add_argument("--epochs",type=int,default=50)
    ap.add_argument("--patience",type=int,default=10)
    ap.add_argument("--batch-size",type=int,default=16)
    ap.add_argument("--mu",type=int,default=3,help="multiplicador do batch não rotulado")
    ap.add_argument("--threshold",type=float,default=0.95)
    ap.add_argument("--lambda-u",type=float,default=1.0)
    ap.add_argument("--lr-backbone",type=float,default=1e-4)
    ap.add_argument("--lr-head",type=float,default=1e-3)
    ap.add_argument("--image-size",type=int,default=224)
    ap.add_argument("--num-workers",type=int,default=2)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--device",default="auto")
    args=ap.parse_args()
    seed_all(args.seed)
    device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available() else
                        ("mps" if args.device=="auto" and hasattr(torch.backends,"mps") and torch.backends.mps.is_available()
                         else ("cpu" if args.device=="auto" else args.device)))
    class_names, labeled_ds, val_ds, unlabeled_ds=load_fixmatch_datasets(args.data_dir,args.image_size)
    # O dataset rotulado para treinamento recebe augmentação moderada.
    # Augmentation supervisionada moderada, consistente com a Parte 0.
    from src.dataset import make_transforms
    labeled_ds.transform=make_transforms(args.image_size)[0]
    pin=device.type=="cuda"
    labeled_loader=DataLoader(labeled_ds,batch_size=args.batch_size,shuffle=True,
                              num_workers=args.num_workers,pin_memory=pin,drop_last=True)
    unl_loader=DataLoader(unlabeled_ds,batch_size=args.batch_size*args.mu,shuffle=True,
                          num_workers=args.num_workers,pin_memory=pin,drop_last=True)
    val_loader=DataLoader(val_ds,batch_size=args.batch_size*2,shuffle=False,num_workers=args.num_workers)
    model=build_model("resnet18",len(class_names),pretrained=True).to(device)
    nparams=count_parameters(model)
    if nparams>12_000_000: raise RuntimeError(f"Modelo excede 12M: {nparams}")
    optimizer=torch.optim.AdamW([
        {"params":[p for n,p in model.named_parameters() if not n.startswith("fc.")],"lr":args.lr_backbone},
        {"params":model.fc.parameters(),"lr":args.lr_head}],weight_decay=1e-4)
    criterion=nn.CrossEntropyLoss()
    criterion_per_example=nn.CrossEntropyLoss(reduction="none")
    # Prior uniforme: o orçamento rotulos_5 tem exatamente cinco exemplos por classe.
    target_prior=torch.full((len(class_names),),1/len(class_names),device=device)
    marginal=torch.full_like(target_prior,1/len(class_names))
    history=[]; best=-1.; patience=0
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for epoch in range(1,args.epochs+1):
        model.train(); lit=iter(labeled_loader); total=0.; sup_sum=0.; unsup_sum=0.; mask_sum=0.; steps=0
        for weak,strong,_ in unl_loader:
            try: x_l,y_l,_=next(lit)
            except StopIteration:
                lit=iter(labeled_loader); x_l,y_l,_=next(lit)
            x_l=x_l.to(device); y_l=y_l.to(device); weak=weak.to(device); strong=strong.to(device)
            logits_l=model(x_l); loss_x=criterion(logits_l,y_l)
            with torch.no_grad():
                model.eval()
                probs=torch.softmax(model(weak),dim=1)
                if args.method=="aligned":
                    marginal=0.999*marginal+0.001*probs.mean(0)
                    probs=probs*(target_prior/(marginal+1e-6))
                    probs=probs/probs.sum(1,keepdim=True).clamp_min(1e-8)
                confidence,pseudo=probs.max(1)
                mask=(confidence>=args.threshold).float()
                model.train()
            logits_u=model(strong)
            per_example_u=criterion_per_example(logits_u,pseudo)
            loss_u=(per_example_u*mask).mean()
            loss=loss_x+args.lambda_u*loss_u
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            sup_sum+=loss_x.item(); unsup_sum+=loss_u.item(); mask_sum+=mask.mean().item(); steps+=1
        val=evaluate(model,val_loader,device,len(class_names))
        row={"epoch":epoch,"train_supervised_loss":sup_sum/max(steps,1),
             "train_unsupervised_loss":unsup_sum/max(steps,1),
             "pseudo_label_acceptance":mask_sum/max(steps,1),
             "val_loss":val["loss"],"val_f1_macro":val["f1_macro"],"val_accuracy":val["accuracy"]}
        history.append(row); pd.DataFrame(history).to_csv(args.output_dir/"history.csv",index=False)
        print(f"[{args.method}] epoch {epoch:02d} | sup={row['train_supervised_loss']:.4f} "
              f"| unsup={row['train_unsupervised_loss']:.4f} | aceitação={row['pseudo_label_acceptance']:.3f} "
              f"| val F1={val['f1_macro']:.4f} | val acc={val['accuracy']:.4f}")
        if val["f1_macro"]>best+1e-8:
            best=val["f1_macro"]; patience=0
            torch.save({"state_dict":model.state_dict(),"class_names":class_names,"epoch":epoch,
                        "best_val_f1_macro":best,"method":args.method,"threshold":args.threshold,
                        "num_parameters":nparams,"seed":args.seed,"image_size":args.image_size},
                       args.output_dir/"best_model.pth")
        else: patience+=1
        if patience>=args.patience:
            print(f"Early stopping após {args.patience} épocas sem melhora."); break
    hist=pd.DataFrame(history)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    ax[0].plot(hist.epoch,hist.train_supervised_loss,label="Supervisionada")
    ax[0].plot(hist.epoch,hist.train_unsupervised_loss,label="Não supervisionada")
    ax[0].set_title("Loss de treino"); ax[0].set_xlabel("Época"); ax[0].legend(); ax[0].grid(alpha=.2)
    ax[1].plot(hist.epoch,hist.val_f1_macro,label="F1 macro validação")
    ax[1].set_title("Validação"); ax[1].set_xlabel("Época"); ax[1].legend(); ax[1].grid(alpha=.2)
    fig.tight_layout(); fig.savefig(args.output_dir/"training_curves.png",dpi=180); plt.close(fig)
    print(f"Melhor F1 val={best:.4f}; checkpoint={args.output_dir/'best_model.pth'}")
    print(f"Parâmetros: {nparams:,}; dispositivo: {device}; não rotuladas: {len(unlabeled_ds)}")

if __name__=="__main__": main()
