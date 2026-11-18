# PA3

## Ambiente

Requisitos:

- Python 3.10 ou superior;
- PyTorch e torchvision;
- pandas, NumPy, Pillow, scikit-learn e matplotlib.

Crie um ambiente virtual e instale as dependências:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Extraia `data/pa3_dados.zip` em `data/` antes do treinamento. A estrutura esperada é:

```text
data/pa3_dados/
  classes.csv
  images/
  labels/
```

Os arquivos de dados e o ZIP não são versionados. `*.pth` também não deve ser enviado ao repositório Git.

## Comando de treinamento

Treine o piso supervisionado da Parte 0:

```powershell
python -m src.train_piso `
  --data-dir data/pa3_dados `
  --output-dir outputs/parte0 `
  --models resnet18
```

Treine o FixMatch aligned da Parte 2 com os orçamentos fornecidos:

```powershell
python -m src.parte2 `
  --data-dir data/pa3_dados `
  --output-dir outputs/parte2 `
  --seeds-5 42 100 200
```

Os experimentos usam os CSVs `labels/rotulos_1.csv`, `labels/rotulos_5.csv` e
`labels/rotulos_20.csv`. Os resultados são escritos em `outputs/parte2`.

## Comando de avaliação

Avalie os modelos da Parte 0 no conjunto público:

```powershell
python -m src.evaluate `
  --data-dir data/pa3_dados `
  --output-dir outputs/parte0 `
  --models both
```

A avaliação da Parte 2 já ocorre ao final de `python -m src.parte2`. Ela gera,
por exemplo:

- `outputs/parte2/curva_orcamento_test_public_final.png`;
- `outputs/parte2/ganho_por_classe_orcamento_5.png`;
- `outputs/parte2/analise_resultados.md`.

## Predição em pasta

O modelo padrão é o checkpoint FixMatch aligned da Parte 1:
`outputs/parte1/fixmatch_aligned/best_model.pth`.

```powershell
python predict.py --imagens PASTA --saida preds.csv
```

Opções úteis:

```powershell
python predict.py `
  --imagens PASTA `
  --saida preds.csv `
  --checkpoint outputs/parte1/fixmatch_aligned/best_model.pth `
  --data-dir data/pa3_dados
```

A saída contém exatamente as colunas `image_id,label`. Os nomes em `label` vêm de
`data/pa3_dados/classes.csv`. O comando percorre subpastas, usa apenas CPU e não
baixa pesos ou dados da internet. Ao finalizar, imprime o número de parâmetros.

## Parte 3: galeria de erros e acertos

```powershell
python -m src.parte3_gallery `
  --data-dir data/pa3_dados `
  --checkpoint outputs/parte1/fixmatch_aligned/best_model.pth `
  --output-dir outputs/parte3
```

A saída inclui galerias com e sem Grad-CAM, imagens individuais e um diagnóstico
completo em `outputs/parte3/diagnostico_completo.md`.