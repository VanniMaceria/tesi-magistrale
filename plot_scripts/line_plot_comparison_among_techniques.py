import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIGURAZIONE ---
ROOT_PATH = r"C:\Users\Lenovo\Desktop\Magistrale\Tesi\progetto\simulation_results\FEMNIST\non-iid"
TECNICHE = ["baseline", "distillation", "ordered_dropout", "quantization"]
# Cartella di output richiesta
OUTPUT_ROOT = os.path.join(ROOT_PATH, "plots", "line_plots")

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.dpi': 300, 'axes.labelcolor': 'black'})

def load_data_all_techniques(entity_name):
    data = []
    for t in TECNICHE:
        sub_path = "num_clients_10" if "Client" in entity_name else ""
        filename = f"{entity_name.lower()}.csv" if "Client" in entity_name else "server_aggregate.csv"
        path = os.path.join(ROOT_PATH, t, sub_path, filename)
        
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['Tecnica'] = t
            data.append(df)
    return pd.concat(data) if data else None

def generate_line_plots(df, entity_name):
    entity_dir = os.path.join(OUTPUT_ROOT, entity_name)
    os.makedirs(entity_dir, exist_ok=True)
    
    metriche = ['accuracy', 'loss', 'energia(J)', 'banda(MB)', 'flops_inferenza']
    
    for m in metriche:
        if m not in df.columns: continue
        plt.figure(figsize=(10, 6))
        # Linea = Media, Ombra = Deviazione Standard
        sns.lineplot(data=df, x='round', y=m, hue='Tecnica', errorbar='sd', linewidth=2)
        
        plt.title(f"Evoluzione Temporale {m.upper()} - {entity_name}")
        plt.xlabel("Round")
        plt.ylabel(m)
        plt.legend(title='Tecniche', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        plt.savefig(os.path.join(entity_dir, f"line_compare_{m.split('(')[0]}.png"))
        plt.close()

def main():
    entities = ["Server_Global"] + [f"Client_{i}" for i in range(10)]
    for entity in entities:
        print(f"Generando Line Plots in plots/line_plots/{entity}...")
        df = load_data_all_techniques(entity)
        if df is not None:
            generate_line_plots(df, entity)

if __name__ == "__main__":
    main()