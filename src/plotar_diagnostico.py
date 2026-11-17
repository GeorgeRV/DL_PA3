import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plotar_diagnostico(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Gráfico da Média Geral (Macro)
    media_df = df.groupby('threshold')[['precision', 'coverage']].mean().reset_index()
    
    plt.figure(figsize=(8, 5))
    plt.plot(media_df['threshold'], media_df['precision'], marker='o', label='Precisão Média', color='blue')
    plt.plot(media_df['threshold'], media_df['coverage'], marker='s', label='Cobertura Média', color='orange')
    plt.title('Trade-off Geral: Precisão vs Cobertura (Média Macro)')
    plt.xlabel('Limiar de Confiança (Threshold)')
    plt.ylabel('Métrica')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(output_dir / 'diagnostico_geral.png', dpi=150)
    plt.close()

    # 2. Gráfico Comparativo: Majoritária vs Minoritária
    classe_major = 'dog'
    classe_minor = 'horse'

    df_major = df[df['class_name'] == classe_major]
    df_minor = df[df['class_name'] == classe_minor]

    fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    
    # Plot Majoritária
    ax[0].plot(df_major['threshold'], df_major['precision'], marker='o', label='Precisão', color='blue')
    ax[0].plot(df_major['threshold'], df_major['coverage'], marker='s', label='Cobertura', color='orange')
    ax[0].set_title(f'Classe Majoritária: {classe_major}')
    ax[0].set_xlabel('Limiar (Threshold)')
    ax[0].set_ylabel('Métrica')
    ax[0].grid(alpha=0.3)
    ax[0].legend()

    # Plot Minoritária
    ax[1].plot(df_minor['threshold'], df_minor['precision'], marker='o', label='Precisão', color='blue')
    ax[1].plot(df_minor['threshold'], df_minor['coverage'], marker='s', label='Cobertura', color='orange')
    ax[1].set_title(f'Classe Minoritária: {classe_minor}')
    ax[1].set_xlabel('Limiar (Threshold)')
    ax[1].grid(alpha=0.3)
    ax[1].legend()

    plt.savefig(output_dir / 'diagnostico_comparativo.png', dpi=150)
    plt.close()
    print(f"Gráficos salvos em: {output_dir}")

if __name__ == '__main__':
    # Caminho exato para o CSV gerado pelo seu diagnose_pseudo_labels.py
    csv_diagnostico = 'outputs/parte1/diagnostics_standard/pseudo_label_precision_coverage_by_class.csv'
    plotar_diagnostico(csv_diagnostico, 'outputs/parte1/diagnostics_standard')