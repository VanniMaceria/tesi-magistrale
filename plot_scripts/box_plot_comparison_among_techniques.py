import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIGURAZIONE ---
ROOT_PATH = r"C:\Users\Lenovo\Desktop\Magistrale\Tesi\progetto\simulation_results\FEMNIST\non-iid"
TECNICHE = ["baseline", "distillation", "ordered_dropout", "quantization"]
OUTPUT_ROOT = os.path.join(ROOT_PATH, "plots", "box_plots")

# Mappatura nomi tecniche e metriche in inglese
TECHNIQUE_LABELS = {
    "baseline": "Baseline",
    "distillation": "FKD",
    "ordered_dropout": "OD",
    "quantization": "QAT"
}

METRIC_LABELS = {
    'accuracy': 'Accuracy',
    'loss': 'Loss',
    'energia(J)': 'Energy (J)',
    'banda(MB)': 'Bandwidth (MB)',
    'flops_inferenza': 'Inference FLOPs'
}

# Configurazione stile e font maggiorati
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'figure.dpi': 300,
    'axes.labelcolor': 'black'
})

def load_final_data(entity_name):
    data = []
    for t in TECNICHE:
        sub_path = "num_clients_10" if "Client" in entity_name else ""
        filename = f"{entity_name.lower()}.csv" if "Client" in entity_name else "server_aggregate.csv"
        path = os.path.join(ROOT_PATH, t, sub_path, filename)
        
        if os.path.exists(path):
            df = pd.read_csv(path)
            last_round = df['round'].max()
            df_final = df[df['round'] == last_round].copy()
            df_final['Tecnica'] = TECHNIQUE_LABELS.get(t, t)
            data.append(df_final)
    return pd.concat(data, ignore_index=True) if data else None

def generate_box_plots(df, entity_name):
    entity_dir = os.path.join(OUTPUT_ROOT, entity_name)
    os.makedirs(entity_dir, exist_ok=True)
    
    metriche = ['accuracy', 'loss', 'energia(J)', 'banda(MB)', 'flops_inferenza']
    
    for m in metriche:
        if m not in df.columns: 
            continue
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.boxplot(data=df, x='Tecnica', y=m, hue='Tecnica', palette="Set2", legend=False, ax=ax)
        sns.swarmplot(data=df, x='Tecnica', y=m, color="black", alpha=0.5, ax=ax)
        
        # Linee verticali bianche sulla griglia
        ax.grid(axis='x', color='white', linestyle='-', linewidth=1.5)
        
        ax.set_xlabel("Technique")
        ax.set_ylabel(METRIC_LABELS.get(m, m))
        
        plt.tight_layout()
        plt.savefig(os.path.join(entity_dir, f"box_compare_{m.split('(')[0]}.png"))
        plt.close()

def main():
    entities = ["Server_Global"] + [f"Client_{i}" for i in range(10)]
    for entity in entities:
        print(f"Generando Box Plots in plots/box_plots/{entity}...")
        df = load_final_data(entity)
        if df is not None:
            generate_box_plots(df, entity)

if __name__ == "__main__":
    main()