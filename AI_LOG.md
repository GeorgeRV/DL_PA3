# Registro de apoio de IA

Este arquivo resume como ferramentas de IA foram utilizadas como apoio técnico
ao desenvolvimento do PA3.

## 1. Organização do projeto

**Prompt/objetivo:** organizar o fluxo do trabalho e identificar os arquivos
necessários para as Partes 0, 1, 2 e 3.

**Como ajudou:** a IA auxiliou na leitura da estrutura do repositório, na
identificação dos scripts existentes e na escolha dos checkpoints e comandos
adequados para cada etapa.

## 2. Curva de orçamento

**Prompt/objetivo:** corrigir os gráficos da Parte 2 e analisar os resultados
dos orçamentos 1, 5 e 20.

**Como ajudou:** a IA auxiliou na consolidação dos resultados do piso e do
FixMatch aligned, no cálculo de média e desvio para as três seeds do orçamento
5, na inclusão da linha do teto e na geração dos gráficos de ganho por classe.

## 3. Análise dos resultados

**Prompt/objetivo:** interpretar a curva de orçamento e os ganhos por classe.

**Como ajudou:** a IA auxiliou na organização dos resultados em uma análise
comparativa, destacando o comportamento do método em cada orçamento, as classes
com maiores ganhos e as classes que apresentaram pouca ou nenhuma melhoria.

## 5. Predição e documentação

**Prompt/objetivo:** disponibilizar uma interface de predição em CPU e
documentar o uso do projeto.

**Como ajudou:** a IA auxiliou na criação do `predict.py`, que gera um CSV com
`image_id,label`, usa os nomes de `classes.csv`, informa o número de parâmetros
e funciona sem acesso à internet. Também auxiliou na atualização do README,
dos comandos de execução e das regras do `.gitignore`.
