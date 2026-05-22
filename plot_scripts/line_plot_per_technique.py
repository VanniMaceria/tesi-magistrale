import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIGURAZIONE PERCORSI ---
BASE_PATH = r"C:\Users\Lenovo\Desktop\Magistrale\Tesi\progetto\simulation_results\MNIST\iid\distillation"
SERVER_FILE = os.path.join(BASE_PATH, "server_aggregate.csv")
CLIENTS_DIR = os.path.join(BASE_PATH, "num_clients_10")
OUTPUT_ROOT = os.path.join(BASE_PATH, "line_plots")

# Stile professionale
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})

def generate_entity_plots(df, entity_name):
    """Genera e salva i line plot organizzati in una cartella specifica per l'entità."""
    
    entity_dir = os.path.join(OUTPUT_ROOT, entity_name)
    os.makedirs(entity_dir, exist_ok=True)

    # --- 1. GRAFICO COMBINATO: ACCURACY & LOSS ---
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Asse Sinistro: Accuracy
    ax1.set_xlabel('Communication Round')
    ax1.set_ylabel('Accuracy', color='tab:green', fontweight='bold')
    sns.lineplot(data=df, x='round', y='accuracy', errorbar='sd', color='tab:green', ax=ax1, linewidth=2.5, label='Accuracy')
    ax1.tick_params(axis='y')
    
    # Asse Destro: Loss
    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss', color='tab:blue', fontweight='bold')
    sns.lineplot(data=df, x='round', y='loss', errorbar='sd', color='tab:blue', ax=ax2, linewidth=2, linestyle='--', label='Loss')
    ax2.tick_params(axis='y')
    
    plt.title(f"Performance Trend: {entity_name}\n(Mean ± SD across experiments)")
    fig.tight_layout()
    
    # Unione legende di entrambi gli assi
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='center right')
    
    plt.savefig(os.path.join(entity_dir, "performance_combined_trend.png"))
    plt.close()

    # --- 2. ALTRE METRICHE (Energia, Banda, FLOPs) ---
    altre_metriche = [
        ('energia(J)', 'tab:red'),
        ('banda(MB)', 'tab:orange'),
        ('flops_inferenza', 'tab:purple')
    ]

    for metrica, colore in altre_metriche:
        if metrica not in df.columns: continue
            
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='round', y=metrica, errorbar='sd', color=colore, linewidth=2)
        
        plt.title(f"Temporal {metrica.upper()} - {entity_name}")
        plt.xlabel("Communication Round")
        plt.ylabel(metrica)
        plt.tight_layout()
        
        clean_name = metrica.replace('(', '').replace(')', '').split('_')[0]
        plt.savefig(os.path.join(entity_dir, f"{clean_name}_trend.png"))
        plt.close()

def main():
    # Processa il Server
    if os.path.exists(SERVER_FILE):
        print("Elaborazione Server_Global...")
        df_s = pd.read_csv(SERVER_FILE)
        generate_entity_plots(df_s, "Server_Global")

    # Processa i Client
    for i in range(10):
        client_file = os.path.join(CLIENTS_DIR, f"client_{i}.csv")
        if os.path.exists(client_file):
            print(f"Elaborazione Client_{i}...")
            df_c = pd.read_csv(client_file)
            generate_entity_plots(df_c, f"Client_{i}")

    print(f"\nCompletato! Grafici salvati per ogni client in: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()